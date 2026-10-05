"""Benchmark circuits: loading QASM / QPY files and generating MQT Bench circuits.

The paper uses the MQT Bench *target-independent* ("indep") Qiskit circuits with
2-127 qubits (1,510 circuits after removing circuits that fail to load/compile or
run longer than 10 minutes). Circuit files are named ``<family>_indep_qiskit_<n>``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Iterator

from qiskit import QuantumCircuit, qpy

log = logging.getLogger(__name__)

CIRCUIT_SUFFIXES = (".qasm", ".qpy")
_FAMILY_RE = re.compile(r"^(?P<family>.+?)_(indep|alg|nativegates|mapped)(_.*)?$")


def circuit_name(path: str | Path) -> str:
    return Path(path).stem


def family_of(name: str) -> str:
    """Algorithm family of a benchmark circuit, e.g. ``qwalk-noancilla_indep_qiskit_9`` -> ``qwalk-noancilla``."""
    name = Path(name).stem if name.endswith(CIRCUIT_SUFFIXES) else name
    m = _FAMILY_RE.match(name)
    if m:
        return m.group("family")
    # fall back to "<family>_<n>"
    return re.sub(r"_\d+$", "", name)


def load_circuit(path: str | Path) -> QuantumCircuit:
    """Load a circuit from an OpenQASM 2/3 or QPY file."""
    path = Path(path)
    if path.suffix == ".qpy":
        with open(path, "rb") as fh:
            circ = qpy.load(fh)[0]
    else:
        text = path.read_text()
        if re.search(r"OPENQASM\s+3", text):
            from qiskit import qasm3

            circ = qasm3.loads(text)
        else:
            # from_qasm_str uses the legacy custom-instruction table, which understands
            # the qelib1.inc gates emitted by older Qiskit/MQT Bench versions (u0, rccx, c3x, ...).
            circ = QuantumCircuit.from_qasm_str(text)
    circ.name = path.stem
    return circ


def save_circuit(circ: QuantumCircuit, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        qpy.dump(circ, fh)


def iter_circuit_files(
    directory: str | Path,
    min_qubits: int = 2,
    max_qubits: int = 127,
    names: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield circuit files in ``directory`` (sorted). The qubit filter reads only the QASM header."""
    directory = Path(directory)
    wanted = set(names) if names is not None else None
    for path in sorted(directory.iterdir()):
        if path.suffix not in CIRCUIT_SUFFIXES:
            continue
        if wanted is not None and path.stem not in wanted:
            continue
        n = _num_qubits_hint(path)
        if n is not None and not (min_qubits <= n <= max_qubits):
            continue
        yield path


def _num_qubits_hint(path: Path) -> int | None:
    m = re.search(r"_(\d+)$", path.stem)
    if m:
        return int(m.group(1))
    return None


def generate_mqt_circuits(
    out_dir: str | Path,
    benchmarks: Iterable[str] | None = None,
    min_qubits: int = 2,
    max_qubits: int = 127,
    step: int = 1,
) -> list[Path]:
    """Generate target-independent MQT Bench circuits as QPY files.

    Uses the installed ``mqt.bench`` (>= 2.0). Benchmarks that cannot be generated
    at a given size are skipped. Note that newer MQT Bench releases contain a
    different benchmark list than the v1.0 set used in the paper; to reproduce the
    paper exactly use ``qetime fetch-paper-data`` instead.
    """
    from mqt.bench import BenchmarkLevel, get_benchmark
    from mqt.bench.benchmarks import get_available_benchmark_names

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(benchmarks) if benchmarks else get_available_benchmark_names()
    written: list[Path] = []
    for bench in names:
        for n in range(min_qubits, max_qubits + 1, step):
            target = out_dir / f"{bench}_indep_qiskit_{n}.qpy"
            if target.exists():
                written.append(target)
                continue
            try:
                circ = get_benchmark(benchmark=bench, level=BenchmarkLevel.INDEP, circuit_size=n)
            except Exception as exc:  # unsupported size, missing extras, ...
                log.debug("skip %s n=%d: %s", bench, n, exc)
                continue
            if circ.num_qubits != n:
                # some benchmarks add ancillas; keep the actual width in the name
                target = out_dir / f"{bench}_indep_qiskit_{circ.num_qubits}.qpy"
                if circ.num_qubits > max_qubits or target.exists():
                    continue
            save_circuit(circ, target)
            written.append(target)
            log.info("generated %s", target.name)
    return written
