"""Dataset assembly: featurize (circuit, backend) pairs, attach measured times, scale, convert to PyG.

A dataset file is a pickle with ``{"samples": [Sample, ...], "meta": {...}}``.
"""

from __future__ import annotations

import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from . import backends as bk
from .circuits import family_of, load_circuit, save_circuit
from .features import (
    DEFAULT_MAX_QUBITS,
    GLOBAL_FEATURE_NAMES,
    NUM_COHERENCE,
    NUM_NODE_TYPES,
    OPENQASM_GATES,
    CircuitGraph,
    circuit_graph,
    global_features,
    initial_layout_indices,
    node_dim,
    node_feature_groups,
    simple_stats,
)

log = logging.getLogger(__name__)

GRAPH_SOURCES = ("compiled", "logical")


@dataclass
class Sample:
    circuit: str
    backend: str
    family: str
    global_raw: np.ndarray
    graph: CircuitGraph
    stats: dict = field(default_factory=dict)
    y: float = float("nan")  # execution time in seconds


# --------------------------------------------------------------------------- measured times
def load_times(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Read execution-time CSVs into ``[circuit, backend, time_s]`` (averaged over repeats).

    Accepts the files written by ``qetime measure-sim`` / ``qetime hw-collect`` and the
    CSVs of the paper's replication package (``quantum_circuit|circuit_name, time_taken, device``).
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        circ = cols.get("circuit") or cols.get("quantum_circuit") or cols.get("circuit_name")
        backend = cols.get("backend") or cols.get("device")
        t = cols.get("time_s") or cols.get("time_taken") or cols.get("actual_s")
        if not (circ and backend and t):
            raise ValueError(f"{p}: cannot find circuit/backend/time columns in {list(df.columns)}")
        if "status" in cols:
            df = df[df[cols["status"]] == "ok"]
        out = pd.DataFrame(
            {
                "circuit": df[circ].astype(str).str.replace(r"\.(qasm|qpy)$", "", regex=True),
                "backend": df[backend].astype(str).map(bk.canonical_name),
                "time_s": pd.to_numeric(df[t], errors="coerce"),
            }
        )
        frames.append(out.dropna())
    times = pd.concat(frames, ignore_index=True)
    return times.groupby(["circuit", "backend"], as_index=False)["time_s"].mean()


# --------------------------------------------------------------------------- featurization
def compiled_path(cache_dir: str | Path, circuit: str, backend: str) -> Path:
    return Path(cache_dir) / bk.canonical_name(backend) / f"{circuit}.qpy"


def get_compiled(circ, circuit: str, backend_name: str, cache_dir: str | Path | None, opt_level: int, seed: int):
    if cache_dir is not None:
        path = compiled_path(cache_dir, circuit, backend_name)
        if path.exists():
            return load_circuit(path)
    compiled = bk.transpile_for(circ, bk.featurization_backend(backend_name), opt_level, seed)
    if cache_dir is not None:
        save_circuit(compiled, compiled_path(cache_dir, circuit, backend_name))
    return compiled


def featurize(
    circuit_file: str | Path,
    backend_name: str,
    graph_source: str = "compiled",
    compiled_dir: str | Path | None = None,
    opt_level: int = 1,
    seed: int = 1234,
) -> Sample:
    """Global + graph features of one circuit for one backend.

    graph_source="compiled": DAG of the circuit compiled for the backend (physical
        qubits, native gates); this is the representation described in the paper.
    graph_source="logical": DAG of the target-independent circuit with the backend's
        T1/T2 looked up by logical qubit index (what the authors' released code does).
    """
    if graph_source not in GRAPH_SOURCES:
        raise ValueError(f"graph_source must be one of {GRAPH_SOURCES}")
    circuit_file = Path(circuit_file)
    name = circuit_file.stem
    qc = load_circuit(circuit_file)
    backend = bk.featurization_backend(backend_name)
    coherence = bk.qubit_coherence(backend)
    if graph_source == "compiled":
        compiled = get_compiled(qc, name, backend_name, compiled_dir, opt_level, seed)
        graph = circuit_graph(compiled, coherence)
    else:
        graph = circuit_graph(qc, coherence)
    return Sample(
        circuit=name,
        backend=bk.canonical_name(backend_name),
        family=family_of(name),
        global_raw=global_features(qc),
        graph=graph,
        stats=simple_stats(qc),
    )


def _featurize_task(args):
    try:
        return featurize(*args), None
    except Exception as exc:  # keep going; report failures
        return None, f"{Path(args[0]).stem}@{args[1]}: {type(exc).__name__}: {exc}"


def build_samples(
    circuit_files: Sequence[str | Path],
    backend_names: Sequence[str],
    graph_source: str = "compiled",
    compiled_dir: str | Path | None = None,
    opt_level: int = 1,
    seed: int = 1234,
    workers: int | None = None,
    times: pd.DataFrame | None = None,
) -> list[Sample]:
    """Featurize every (circuit, backend) pair, optionally restricted to pairs with a measured time."""
    wanted = None
    if times is not None:
        wanted = set(zip(times.circuit, times.backend))
    tasks = []
    for f in circuit_files:
        for b in backend_names:
            if wanted is not None and (Path(f).stem, bk.canonical_name(b)) not in wanted:
                continue
            tasks.append((str(f), b, graph_source, str(compiled_dir) if compiled_dir else None, opt_level, seed))
    workers = workers or max(1, (os.cpu_count() or 2))
    samples: list[Sample] = []
    failures = []
    if workers == 1:
        results = (_featurize_task(t) for t in tasks)
        for i, (s, err) in enumerate(results, 1):
            (samples.append(s) if s else failures.append(err))
            if i % 100 == 0:
                log.info("featurized %d/%d", i, len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_featurize_task, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                s, err = fut.result()
                (samples.append(s) if s else failures.append(err))
                if i % 100 == 0:
                    log.info("featurized %d/%d", i, len(tasks))
    for err in failures:
        log.warning("featurization failed: %s", err)
    samples.sort(key=lambda s: (s.circuit, s.backend))
    if times is not None:
        attach_times(samples, times)
    return samples


def attach_times(samples: list[Sample], times: pd.DataFrame) -> None:
    lookup = {(c, b): t for c, b, t in zip(times.circuit, times.backend, times.time_s)}
    for s in samples:
        s.y = float(lookup.get((s.circuit, s.backend), float("nan")))


def save_samples(samples: list[Sample], path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"samples": samples, "meta": meta or {}}, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_samples(path: str | Path) -> tuple[list[Sample], dict]:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    return obj["samples"], obj.get("meta", {})


def filter_samples(samples: list[Sample], backends: Iterable[str] | None = None, labelled: bool = True) -> list[Sample]:
    keep = {bk.canonical_name(b) for b in backends} if backends else None
    out = []
    for s in samples:
        if keep is not None and s.backend not in keep:
            continue
        if labelled and not np.isfinite(s.y):
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------- scaling
_COUNT_COLS = set(OPENQASM_GATES) | {"num_qubits", "depth"}


def _safe_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Columns that are constant in the training data keep unit scale.

    Without this, a gate that never occurs in the training split (e.g. when a whole
    algorithm family is held out) would be divided by ~0 and explode at test time.
    (Such global columns are additionally zeroed, see ``FeatureScaler.g_const``.)
    """
    return np.where(std < eps, 1.0, std)


class FeatureScaler:
    """Selects non-zero global columns and standardizes inputs and targets.

    * global features: columns that are zero for every fitted sample are dropped
      (44 gates + 7 -> 41 dims on the paper's data); count-like columns are
      optionally log1p-transformed; then z-scored.
    * node features: binary columns are kept; the continuous T1/T2 and node-index
      columns are z-scored.
    * target: z-scored seconds (``target="standard"``) or z-scored log seconds
      (``target="log"``); metrics are always reported in seconds.
    """

    def __init__(self, max_qubits: int = DEFAULT_MAX_QUBITS, log_counts: bool = True, target: str = "standard"):
        if target not in ("standard", "log"):
            raise ValueError("target must be 'standard' or 'log'")
        self.max_qubits = max_qubits
        self.log_counts = log_counts
        self.target = target
        self.global_cols: np.ndarray | None = None

    # -- fitting
    def fit(self, samples: Sequence[Sample], global_cols_from: Sequence[Sample] | None = None) -> "FeatureScaler":
        G = np.stack([s.global_raw for s in (global_cols_from or samples)])
        self.global_cols = np.flatnonzero(np.abs(G).sum(axis=0) > 0)
        g = self._prep_global(np.stack([s.global_raw for s in samples]))
        self.g_mean = g.mean(axis=0)
        self.g_std = _safe_std(g.std(axis=0))
        # columns that never vary in training carry no learnable signal -> fed as 0
        self.g_const = g.std(axis=0) < 1e-6

        coh = np.concatenate([s.graph.coherence for s in samples])
        pos = np.concatenate([s.graph.position for s in samples])
        cont = np.concatenate([coh, pos[:, None]], axis=1).astype(np.float64)
        self.n_mean = cont.mean(axis=0)
        self.n_std = _safe_std(cont.std(axis=0))

        y = self._y_forward(np.array([s.y for s in samples], dtype=np.float64))
        self.y_mean = float(y.mean())
        self.y_std = float(y.std() + 1e-8)
        return self

    @property
    def global_names(self) -> list[str]:
        return [GLOBAL_FEATURE_NAMES[i] for i in self.global_cols]

    @property
    def global_dim(self) -> int:
        return int(len(self.global_cols))

    @property
    def node_dim(self) -> int:
        return node_dim(self.max_qubits)

    # -- transforms
    def _prep_global(self, G: np.ndarray) -> np.ndarray:
        G = G[:, self.global_cols].astype(np.float64)
        if self.log_counts:
            names = self.global_names
            idx = [i for i, n in enumerate(names) if n in _COUNT_COLS]
            G[:, idx] = np.log1p(G[:, idx])
        return G

    def transform_global(self, raw: np.ndarray) -> np.ndarray:
        g = self._prep_global(np.atleast_2d(raw))
        z = (g - self.g_mean) / self.g_std
        z[:, getattr(self, "g_const", np.zeros(z.shape[1], bool))] = 0.0
        return z.astype(np.float32)

    def transform_graph(self, graph: CircuitGraph) -> np.ndarray:
        x = graph.to_dense(self.max_qubits)
        a = NUM_NODE_TYPES + self.max_qubits
        x[:, a : a + NUM_COHERENCE + 1] = (x[:, a : a + NUM_COHERENCE + 1] - self.n_mean) / self.n_std
        return x

    def _y_forward(self, y: np.ndarray) -> np.ndarray:
        return np.log(y) if self.target == "log" else y

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        return ((self._y_forward(np.asarray(y, dtype=np.float64)) - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, z: np.ndarray) -> np.ndarray:
        v = np.asarray(z, dtype=np.float64) * self.y_std + self.y_mean
        return np.exp(v) if self.target == "log" else v

    # -- (de)serialization
    def state_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_state_dict(cls, state: dict) -> "FeatureScaler":
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        if "g_const" not in state:  # checkpoints written before g_const existed
            obj.g_const = np.asarray(obj.g_std) < 1e-6
        obj.g_std, obj.n_std = _safe_std(obj.g_std), _safe_std(obj.n_std)
        return obj


# --------------------------------------------------------------------------- PyG conversion
def node_column_mask(max_qubits: int, drop: Iterable[str] = ()) -> np.ndarray:
    """Boolean mask over node-feature columns; ``drop`` names groups of :func:`node_feature_groups`."""
    mask = np.ones(node_dim(max_qubits), dtype=bool)
    groups = node_feature_groups(max_qubits)
    for g in drop:
        if g not in groups:
            raise ValueError(f"unknown node feature group '{g}', choose from {list(groups)}")
        mask[groups[g]] = False
    return mask


MAX_ARITY = 5  # largest qelib1 gate (c4x); wider gates keep their first 5 qubits


def to_pyg(samples: Sequence[Sample], scaler: FeatureScaler, with_graph: bool = True) -> list:
    """Convert samples into ``torch_geometric.data.Data`` objects.

    Node features are stored compactly -- ``nt`` (type id), ``nq`` (qubit ids, -1
    padded), ``nc`` (scaled T1/T2 x2 + node index) -- and expanded to the 178-d
    vectors inside the model (:class:`qetime.model.NodeFeatures`).
    """
    import torch
    from torch_geometric.data import Data

    out = []
    for i, s in enumerate(samples):
        y = scaler.transform_y([s.y])[0] if np.isfinite(s.y) else 0.0
        if not with_graph:  # global-feature-only models skip the (large) node tensors
            out.append(Data(g=torch.from_numpy(scaler.transform_global(s.global_raw)),
                            y=torch.tensor([y], dtype=torch.float32), idx=torch.tensor([i]), num_nodes=1))
            continue
        g = s.graph
        nq = np.full((g.num_nodes, MAX_ARITY), -1, dtype=np.int64)
        k = min(MAX_ARITY, g.qubits.shape[1])
        nq[:, :k] = g.qubits[:, :k]
        cont = np.concatenate([g.coherence, g.position[:, None]], axis=1).astype(np.float64)
        nc = ((cont - scaler.n_mean) / scaler.n_std).astype(np.float32)
        out.append(
            Data(
                nt=torch.from_numpy(g.node_type.astype(np.int64)),
                nq=torch.from_numpy(nq),
                nc=torch.from_numpy(nc),
                edge_index=torch.from_numpy(g.edge_index),
                g=torch.from_numpy(scaler.transform_global(s.global_raw)),
                y=torch.tensor([y], dtype=torch.float32),
                idx=torch.tensor([i]),
                num_nodes=g.num_nodes,
            )
        )
    return out
