"""Backends: fake (simulator) backends, IBM Quantum backends, and qubit coherence data.

Name resolution (case-insensitive):

* ``ibm_<name>``                -> real device through ``QiskitRuntimeService`` (needs an account).
* ``sherbrooke`` / ``FakeSherbrooke`` / ``fake_sherbrooke``
                                -> ``qiskit_ibm_runtime.fake_provider.FakeSherbrooke``.
  The same works for every fake backend that ships with ``qiskit-ibm-runtime``
  (``washington`` resolves to ``FakeWashingtonV2``; ``osaka``/``kyoto`` resolve to the
  calibration snapshots of the retired ``ibm_osaka``/``ibm_kyoto`` devices).
"""

from __future__ import annotations

import functools
import logging
import os
import re

import numpy as np

log = logging.getLogger(__name__)


def canonical_name(name: str) -> str:
    """Short lower-case backend name used in datasets: ``FakeSherbrooke`` -> ``sherbrooke``."""
    n = name.strip()
    n = re.sub(r"^(fake_?|ibm_)", "", n, flags=re.IGNORECASE)
    n = re.sub(r"V\d+$", "", n)
    return n.lower()


def is_real(name: str) -> bool:
    return name.lower().startswith("ibm_")


@functools.lru_cache(maxsize=None)
def get_fake_backend(name: str):
    from qiskit_ibm_runtime import fake_provider

    short = canonical_name(name)
    candidates = [f"Fake{short.capitalize()}", f"Fake{short.capitalize()}V2"]
    for cls_name in candidates:
        cls = getattr(fake_provider, cls_name, None)
        if cls is not None:
            return cls()
    available = sorted(n for n in dir(fake_provider) if n.startswith("Fake"))
    raise ValueError(f"Unknown fake backend '{name}'. Available: {', '.join(available)}")


@functools.lru_cache(maxsize=None)
def get_runtime_service(channel: str | None = None, instance: str | None = None):
    from qiskit_ibm_runtime import QiskitRuntimeService

    kwargs = {}
    token = os.environ.get("QISKIT_IBM_TOKEN")
    channel = channel or os.environ.get("QISKIT_IBM_CHANNEL")
    instance = instance or os.environ.get("QISKIT_IBM_INSTANCE")
    if token:
        kwargs["token"] = token
    if channel:
        kwargs["channel"] = channel
    if instance:
        kwargs["instance"] = instance
    try:
        return QiskitRuntimeService(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "No usable IBM Quantum account: set QISKIT_IBM_TOKEN (and QISKIT_IBM_INSTANCE / "
            "QISKIT_IBM_CHANNEL if needed) or save one with QiskitRuntimeService.save_account(...). "
            f"Original error: {exc}"
        ) from exc


def get_backend(name: str):
    """Return a backend object for ``name`` (see module docstring)."""
    if is_real(name):
        return get_runtime_service().backend(name.lower())
    return get_fake_backend(name)


def get_simulator(name: str):
    """Noise-aware Aer simulator built from a fake backend's calibration snapshot."""
    from qiskit_aer import AerSimulator

    return AerSimulator.from_backend(get_fake_backend(name))


@functools.lru_cache(maxsize=None)
def featurization_backend(name: str):
    """Backend whose target/calibration is used to compile + featurize circuits.

    For real devices that are retired (e.g. ``ibm_osaka``) or when no IBM account is
    configured, the bundled calibration snapshot of the same device is used.
    """
    if is_real(name):
        try:
            return get_backend(name)
        except Exception as exc:  # no account / retired device
            log.warning("Could not reach %s (%s); using its fake snapshot instead.", name, exc)
    return get_fake_backend(name)


def qubit_coherence(backend) -> np.ndarray:
    """Array of shape (num_qubits, 2) with [T1, T2] in microseconds (0 when unknown)."""
    n = backend.num_qubits
    out = np.zeros((n, 2), dtype=np.float32)
    props = None
    target = getattr(backend, "target", None)
    if target is not None and getattr(target, "qubit_properties", None):
        props = target.qubit_properties
    for q in range(n):
        p = None
        if props is not None and q < len(props):
            p = props[q]
        else:
            try:
                p = backend.qubit_properties(q)
            except Exception:
                p = None
        if p is None:
            continue
        t1 = getattr(p, "t1", None)
        t2 = getattr(p, "t2", None)
        out[q, 0] = (t1 or 0.0) * 1e6
        out[q, 1] = (t2 or 0.0) * 1e6
    return np.nan_to_num(out)


def transpile_for(circ, backend, optimization_level: int = 1, seed: int = 1234):
    """Compile a target-independent circuit for ``backend`` (target-dependent mapped level)."""
    from qiskit import transpile

    return transpile(circ, backend=backend, optimization_level=optimization_level, seed_transpiler=seed)
