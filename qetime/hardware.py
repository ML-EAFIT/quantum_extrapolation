"""Execution-time collection on IBM Quantum hardware (paper Sections 4.3, 6.1.3).

Queues on real devices take hours, so collection is split into two steps:

1. ``submit``  - compile every selected circuit for the device, submit it ``repeats``
   times with 1024 shots (Sampler V2, job mode) and record the job ids together with
   IBM's own pre-execution estimate (``job.usage_estimation['quantum_seconds']``).
2. ``collect`` - later, fetch each finished job's actual quantum time
   (``job.usage()`` / ``job.metrics()['usage']['quantum_seconds']``), i.e. the time the
   QPU was dedicated to the job, excluding queueing.

Credentials come from the saved Qiskit account or the ``QISKIT_IBM_TOKEN``,
``QISKIT_IBM_CHANNEL`` and ``QISKIT_IBM_INSTANCE`` environment variables.

Note: the devices used in the paper (ibm_osaka, ibm_kyoto) have been retired; use
any currently available device (e.g. ``ibm_fez``, ``ibm_torino``) and build its
dataset with the same backend name.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

FIELDS = ["circuit", "backend", "repeat", "job_id", "status", "estimated_s", "actual_s", "submitted_at", "error"]


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    tmp.replace(path)


def submit(
    circuit_files: Iterable[str | Path],
    backend_name: str,
    out_csv: str | Path,
    shots: int = 1024,
    repeats: int = 3,
    optimization_level: int = 1,
    seed: int = 1234,
) -> Path:
    """Submit circuits to an IBM device; resumable (already-submitted repeats are skipped)."""
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    from .backends import canonical_name, get_backend
    from .circuits import load_circuit

    out_csv = Path(out_csv)
    rows = _read_rows(out_csv)
    have = {(r["circuit"], r["backend"], int(r["repeat"])) for r in rows if r.get("job_id")}
    backend = get_backend(backend_name)
    short = canonical_name(backend_name)
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend, seed_transpiler=seed)
    sampler = SamplerV2(mode=backend)

    for f in circuit_files:
        name = Path(f).stem
        todo = [r for r in range(repeats) if (name, short, r) not in have]
        if not todo:
            continue
        try:
            isa = pm.run(load_circuit(f))
        except Exception as exc:
            rows.append({"circuit": name, "backend": short, "repeat": -1, "status": "error", "error": str(exc)[:300]})
            _write_rows(out_csv, rows)
            continue
        for r in todo:
            job = sampler.run([isa], shots=shots)
            est = ""
            try:
                est = (job.usage_estimation or {}).get("quantum_seconds", "")
            except Exception as exc:  # estimate not available for every plan/channel
                log.debug("no usage estimation for %s: %s", job.job_id(), exc)
            rows.append({"circuit": name, "backend": short, "repeat": r, "job_id": job.job_id(),
                         "status": "submitted", "estimated_s": est, "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            _write_rows(out_csv, rows)
            log.info("submitted %s repeat %d -> %s (IBM estimate %s s)", name, r, job.job_id(), est)
    return out_csv


def _actual_seconds(job) -> float | None:
    try:
        v = job.usage()
        if v is not None:
            return float(v)
    except Exception:
        pass
    try:
        return float(job.metrics()["usage"]["quantum_seconds"])
    except Exception:
        return None


def collect(out_csv: str | Path) -> Path:
    """Fill in the actual quantum seconds of finished jobs in ``out_csv``."""
    from .backends import get_runtime_service

    out_csv = Path(out_csv)
    rows = _read_rows(out_csv)
    service = get_runtime_service()
    for r in rows:
        if r.get("status") not in ("submitted", "running", "queued") or not r.get("job_id"):
            continue
        job = service.job(r["job_id"])
        status = str(job.status())
        status = getattr(job.status(), "name", status)
        if status == "DONE":
            secs = _actual_seconds(job)
            r["status"], r["actual_s"] = ("ok", secs) if secs is not None else ("error", "")
            if not r.get("estimated_s"):
                try:
                    r["estimated_s"] = (job.usage_estimation or {}).get("quantum_seconds", "")
                except Exception:
                    pass
        elif status in ("ERROR", "CANCELLED"):
            r["status"] = "error"
            r["error"] = str(getattr(job, "error_message", lambda: "")())[:300]
        else:
            r["status"] = status.lower()
    _write_rows(out_csv, rows)
    n_ok = sum(r["status"] == "ok" for r in rows)
    log.info("%d/%d jobs finished", n_ok, len(rows))
    return out_csv
