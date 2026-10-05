"""Execution-time measurement on noise-aware simulators (paper Sections 4.3-4.4).

Each circuit is compiled for the simulator's backend (target-dependent mapped
level) and executed for 1024 shots; the execution time is the ``time_taken``
reported by Qiskit for the job. Every circuit is run several times and the mean is
used. Eq. 1 gives the number of repetitions needed for a precision of r% at a
confidence level of 1-alpha::

    n = ceil( (100 * z_{1-alpha/2} * s / (r * mean))^2 )

The paper uses r = 25 %, 95 % confidence (z = 1.960) and 3 repetitions (n <= 3 for
over 90 % of the circuits). Circuits that fail to load/compile/run, or that take
longer than the timeout (10 minutes), are recorded with a non-"ok" status and
excluded from the dataset.
"""

from __future__ import annotations

import csv
import logging
import math
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)

FIELDS = [
    "circuit", "backend", "status", "time_s", "time_std", "n_repeats", "n_required",
    "wall_s", "times", "num_qubits", "error",
]


def required_repeats(times: Iterable[float], z: float = 1.960, r: float = 25.0) -> int:
    """Minimum number of observations for an r% precise mean (Eq. 1, Jain 1991)."""
    t = np.asarray(list(times), dtype=float)
    if t.size < 2 or t.mean() <= 0:
        return int(t.size)
    s = t.std(ddof=1)
    return int(math.ceil((100.0 * z * s / (r * t.mean())) ** 2))


# --------------------------------------------------------------------------- worker process
def _worker_main(backend_name: str, inq: mp.Queue, outq: mp.Queue) -> None:
    import warnings

    warnings.filterwarnings("ignore")
    from .backends import get_simulator
    from .circuits import load_circuit
    from .dataset import get_compiled

    sim = get_simulator(backend_name)
    while True:
        task = inq.get()
        if task is None:
            return
        path, opt_level, seed, shots, compiled_dir = task["path"], task["opt"], task["seed"], task["shots"], task["cache"]
        try:
            qc = load_circuit(path)
            tqc = get_compiled(qc, Path(path).stem, backend_name, compiled_dir, opt_level, seed)
            outq.put(("compiled", qc.num_qubits))
            while True:
                cmd = inq.get()
                if cmd != "run":
                    break
                t0 = time.perf_counter()
                result = sim.run(tqc, shots=shots).result()
                wall = time.perf_counter() - t0
                if not result.success:
                    raise RuntimeError(getattr(result, "status", "simulation failed"))
                outq.put(("rep", float(result.time_taken), wall))
        except Exception as exc:
            # the parent stops talking about this task after an error
            outq.put(("error", f"{type(exc).__name__}: {exc}"[:500]))


class SimulatorWorker:
    """A persistent child process running one simulator; killed and restarted on timeout."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self.ctx = mp.get_context("spawn")
        self._start()

    def _start(self) -> None:
        self.inq = self.ctx.Queue()
        self.outq = self.ctx.Queue()
        self.proc = self.ctx.Process(target=_worker_main, args=(self.backend_name, self.inq, self.outq), daemon=True)
        self.proc.start()

    def restart(self) -> None:
        self.proc.kill()
        self.proc.join()
        self._start()

    def close(self) -> None:
        try:
            self.inq.put(None)
            self.proc.join(timeout=5)
        finally:
            if self.proc.is_alive():
                self.proc.kill()

    def measure(
        self,
        path: str | Path,
        shots: int = 1024,
        min_repeats: int = 3,
        max_repeats: int = 3,
        timeout: float = 600.0,
        compile_timeout: float = 1800.0,
        opt_level: int = 1,
        seed: int = 1234,
        compiled_dir: str | Path | None = None,
        precision: float = 25.0,
        z: float = 1.960,
    ) -> dict:
        row = {"circuit": Path(path).stem, "backend": self.backend_name, "status": "ok", "error": ""}
        self.inq.put({"path": str(path), "opt": opt_level, "seed": seed, "shots": shots,
                      "cache": str(compiled_dir) if compiled_dir else None})
        try:
            msg = self.outq.get(timeout=compile_timeout + 120)  # includes simulator start-up
        except queue.Empty:
            self.restart()
            return {**row, "status": "timeout", "error": "compilation timeout"}
        if msg[0] == "error":
            return {**row, "status": "error", "error": msg[1]}
        row["num_qubits"] = msg[1]
        times, walls = [], []
        target = min_repeats
        while len(times) < target:
            self.inq.put("run")
            try:
                msg = self.outq.get(timeout=timeout)
            except queue.Empty:
                self.restart()
                return {**row, "status": "timeout", "error": f"execution exceeded {timeout:.0f}s",
                        "times": ";".join(f"{t:.6f}" for t in times)}
            if msg[0] == "error":
                return {**row, "status": "error", "error": msg[1]}
            times.append(msg[1])
            walls.append(msg[2])
            if len(times) == target and target < max_repeats:
                target = min(max(target, required_repeats(times, z, precision)), max_repeats)
        self.inq.put("done")
        t = np.asarray(times)
        row.update(
            time_s=float(t.mean()),
            time_std=float(t.std(ddof=1)) if t.size > 1 else 0.0,
            n_repeats=int(t.size),
            n_required=required_repeats(t, z, precision),
            wall_s=float(np.mean(walls)),
            times=";".join(f"{v:.6f}" for v in t),
        )
        return row


# --------------------------------------------------------------------------- driver
def measure_simulators(
    circuit_files: Iterable[str | Path],
    backend_names: Iterable[str],
    out_csv: str | Path,
    compiled_dir: str | Path | None = None,
    **kwargs,
) -> Path:
    """Measure every circuit on every simulator backend, appending rows to ``out_csv`` (resumable)."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str]] = set()
    if out_csv.exists():
        with open(out_csv) as fh:
            done = {(r["circuit"], r["backend"]) for r in csv.DictReader(fh)}
    new_file = not out_csv.exists()
    files = list(circuit_files)
    from .backends import canonical_name

    with open(out_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for backend in backend_names:
            name = canonical_name(backend)
            todo = [f for f in files if (Path(f).stem, name) not in done]
            if not todo:
                continue
            worker = SimulatorWorker(backend)
            try:
                for i, f in enumerate(todo, 1):
                    row = worker.measure(f, compiled_dir=compiled_dir, **kwargs)
                    row["backend"] = name
                    writer.writerow({k: row.get(k, "") for k in FIELDS})
                    fh.flush()
                    log.info("[%s %d/%d] %s %s %s", name, i, len(todo), row["circuit"], row["status"],
                             f"{row.get('time_s', float('nan')):.3f}s" if row["status"] == "ok" else row["error"][:80])
            finally:
                worker.close()
    return out_csv
