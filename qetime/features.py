"""Circuit representation (paper Section 5.1).

Global features (Section 5.1.1)
    Counts of each OpenQASM-2 standard gate (44 candidates; the paper's dataset uses
    34 of them, all-zero columns are dropped when the dataset is assembled), the
    number of qubits, the depth, and the five SupermarQ features: program
    communication, critical depth, entanglement ratio, parallelism and liveness.
    -> 34 + 7 = 41 dimensions on the paper's dataset.

Graph features (Section 5.1.2)
    The circuit is a DAG whose nodes are the initial qubit states, gates and
    measurements; edges follow the qubit wires. Each node gets a 178-d vector:

        [0:46)     one-hot node type (qubit-in, measure, 44 gate types)
        [46:173)   multi-hot qubit position(s) (127 qubits)
        [173:177)  T1, T2 of the first qubit, T1, T2 of the second qubit (microseconds)
        [177]      sequential position (index) of the node

Graphs are stored compactly (:class:`CircuitGraph`) and expanded to dense
matrices only when needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

# Gates of the OpenQASM 2.0 standard header (qelib1.inc), in its order.
OPENQASM_GATES: list[str] = [
    "u3", "u2", "u1", "cx", "id", "u0", "u", "p", "x", "y", "z", "h", "s", "sdg",
    "t", "tdg", "rx", "ry", "rz", "sx", "sxdg", "cz", "cy", "swap", "ch", "ccx",
    "cswap", "crx", "cry", "crz", "cu1", "cp", "cu3", "csx", "cu", "rxx", "rzz",
    "rccx", "rc3x", "c3x", "c3sqrtx", "c4x", "xx_plus_yy", "ecr",
]
NODE_TYPES: list[str] = ["qubit_in", "measure"] + OPENQASM_GATES
NODE_TYPE_INDEX = {name: i for i, name in enumerate(NODE_TYPES)}
# Qiskit spells a few qelib1 gates differently.
GATE_ALIASES = {"mcx": "c3x", "mcx_gray": "c4x", "c3sx": "c3sqrtx", "rcccx": "rc3x", "i": "id"}

NUM_NODE_TYPES = len(NODE_TYPES)  # 46
NUM_COHERENCE = 4
DEFAULT_MAX_QUBITS = 127

SUPERMARQ_NAMES = [
    "program_communication",
    "critical_depth",
    "entanglement_ratio",
    "parallelism",
    "liveness",
]
GLOBAL_FEATURE_NAMES: list[str] = OPENQASM_GATES + ["num_qubits", "depth"] + SUPERMARQ_NAMES

_SKIP_OPS = {"barrier", "delay", "snapshot"}


def node_dim(max_qubits: int = DEFAULT_MAX_QUBITS) -> int:
    return NUM_NODE_TYPES + max_qubits + NUM_COHERENCE + 1


def node_feature_groups(max_qubits: int = DEFAULT_MAX_QUBITS) -> dict[str, slice]:
    """Column ranges of the node-feature components (used for the ablations of RQ2/RQ3)."""
    a = NUM_NODE_TYPES
    b = a + max_qubits
    c = b + NUM_COHERENCE
    return {
        "node_type": slice(0, a),
        "qubit_index": slice(a, b),
        "t1t2": slice(b, c),
        "node_index": slice(c, c + 1),
    }


def _canonical_gate(name: str) -> str:
    return GATE_ALIASES.get(name, name)


# --------------------------------------------------------------------------- global features
def supermarq_features(qc: QuantumCircuit) -> dict[str, float]:
    """SupermarQ feature vector (Tomesh et al., HPCA'22), as used by MQT Predictor."""
    n = qc.num_qubits
    qindex = {q: i for i, q in enumerate(qc.qubits)}
    neighbours: list[set[int]] = [set() for _ in range(n)]
    liveness_sum = 0
    num_gates = 0
    num_multi = 0
    for inst in qc.data:
        name = inst.operation.name
        if name in _SKIP_OPS or name == "measure":
            continue
        idx = [qindex[q] for q in inst.qubits]
        num_gates += 1
        liveness_sum += len(idx)
        if len(idx) > 1:
            num_multi += 1
            for i in idx:
                neighbours[i].update(j for j in idx if j != i)

    depth = qc.depth(lambda x: x.operation.name not in _SKIP_OPS | {"measure"})
    if n > 1:
        program_communication = sum(len(s) for s in neighbours) / (n * (n - 1))
    else:
        program_communication = 0.0
    if num_multi:
        multi_depth = qc.depth(lambda x: len(x.qubits) > 1 and x.operation.name not in _SKIP_OPS)
        critical_depth = multi_depth / num_multi
    else:
        critical_depth = 0.0
    entanglement_ratio = num_multi / num_gates if num_gates else 0.0
    parallelism = ((num_gates / depth - 1) / (n - 1)) if (depth and n > 1) else 0.0
    liveness = liveness_sum / (depth * n) if (depth and n) else 0.0

    clip = lambda v: float(min(max(v, 0.0), 1.0))  # noqa: E731
    return {
        "program_communication": clip(program_communication),
        "critical_depth": clip(critical_depth),
        "entanglement_ratio": clip(entanglement_ratio),
        "parallelism": clip(parallelism),
        "liveness": clip(liveness),
    }


def global_features(qc: QuantumCircuit) -> np.ndarray:
    """Raw global feature vector, ordered as :data:`GLOBAL_FEATURE_NAMES` (51 values)."""
    counts = {}
    for name, c in qc.count_ops().items():
        key = _canonical_gate(name)
        counts[key] = counts.get(key, 0) + c
    vec = [float(counts.get(g, 0)) for g in OPENQASM_GATES]
    vec += [float(qc.num_qubits), float(qc.depth())]
    sm = supermarq_features(qc)
    vec += [sm[k] for k in SUPERMARQ_NAMES]
    return np.asarray(vec, dtype=np.float64)


def simple_stats(qc: QuantumCircuit) -> dict[str, float]:
    """Target-independent statistics used in RQ1 (Tables 5 and 8)."""
    ops = qc.count_ops()
    num_gates = sum(c for k, c in ops.items() if k not in _SKIP_OPS and k != "measure")
    return {
        "depth": float(qc.depth()),
        "num_qubits": float(qc.num_qubits),
        "two_qubit_gates": float(sum(1 for i in qc.data if len(i.qubits) == 2 and i.operation.name not in _SKIP_OPS)),
        "num_gates": float(num_gates),
    }


# --------------------------------------------------------------------------- graph features
@dataclass
class CircuitGraph:
    """Compact DAG representation of a circuit.

    node_type : (N,) int16   index into NODE_TYPES, -1 for gates outside the vocabulary
    qubits    : (N, K) int16 qubit index/indices of the node, -1 padded
    coherence : (N, 4) float32  [T1_a, T2_a, T1_b, T2_b] in microseconds
    position  : (N,) float32 sequential index of the node
    edge_index: (2, E) int64  directed edges along the qubit wires
    """

    node_type: np.ndarray
    qubits: np.ndarray
    coherence: np.ndarray
    position: np.ndarray
    edge_index: np.ndarray

    @property
    def num_nodes(self) -> int:
        return int(self.node_type.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to_dense(self, max_qubits: int = DEFAULT_MAX_QUBITS) -> np.ndarray:
        n = self.num_nodes
        x = np.zeros((n, node_dim(max_qubits)), dtype=np.float32)
        rows = np.arange(n)
        known = self.node_type >= 0
        x[rows[known], self.node_type[known].astype(np.int64)] = 1.0
        for k in range(self.qubits.shape[1]):
            q = self.qubits[:, k].astype(np.int64)
            ok = (q >= 0) & (q < max_qubits)
            x[rows[ok], NUM_NODE_TYPES + q[ok]] = 1.0
        base = NUM_NODE_TYPES + max_qubits
        x[:, base : base + NUM_COHERENCE] = self.coherence
        x[:, -1] = self.position
        return x

    def mean_vector(self, max_qubits: int = DEFAULT_MAX_QUBITS) -> np.ndarray:
        """Mean-pooled node features (fixed-size summary used for greedy sampling)."""
        return self.to_dense(max_qubits).mean(axis=0)


def circuit_graph(
    qc: QuantumCircuit,
    coherence: np.ndarray | None = None,
    layout: list[int] | None = None,
    keep_measurements: bool = True,
) -> CircuitGraph:
    """Convert a circuit into its DAG representation (paper Fig. 4).

    Parameters
    ----------
    qc:
        The circuit. For the paper's set-up this is the circuit *compiled* for the
        backend, so qubit indices are physical qubits.
    coherence:
        (num_device_qubits, 2) array of [T1, T2] in microseconds (see
        :func:`qetime.backends.qubit_coherence`). ``None`` -> zeros.
    layout:
        Optional map from circuit qubit index to device qubit index used to look up
        T1/T2 (needed when ``qc`` is a logical circuit). Identity when omitted.
    keep_measurements:
        Keep measurement nodes (as in Fig. 4).

    Idle wires (qubits without operations) and barriers are dropped.
    """
    qindex = {q: i for i, q in enumerate(qc.qubits)}
    ops = []
    for inst in qc.data:
        name = inst.operation.name
        if name in _SKIP_OPS:
            continue
        if name == "measure" and not keep_measurements:
            continue
        ops.append((name, [qindex[q] for q in inst.qubits]))

    active = sorted({q for _, qs in ops for q in qs})
    arity = max([2] + [len(qs) for _, qs in ops])
    n_nodes = len(active) + len(ops)

    node_type = np.full(n_nodes, -1, dtype=np.int16)
    qubits = np.full((n_nodes, arity), -1, dtype=np.int16)
    coh = np.zeros((n_nodes, NUM_COHERENCE), dtype=np.float32)
    position = np.arange(n_nodes, dtype=np.float32)

    def lookup(q: int) -> tuple[float, float]:
        if coherence is None:
            return 0.0, 0.0
        dq = layout[q] if layout is not None else q
        if dq is None or dq < 0 or dq >= len(coherence):
            return 0.0, 0.0
        return float(coherence[dq, 0]), float(coherence[dq, 1])

    last: dict[int, int] = {}
    src: list[int] = []
    dst: list[int] = []
    node = 0
    for q in active:
        node_type[node] = NODE_TYPE_INDEX["qubit_in"]
        qubits[node, 0] = q
        coh[node, 0:2] = lookup(q)
        last[q] = node
        node += 1
    for name, qs in ops:
        key = "measure" if name == "measure" else _canonical_gate(name)
        node_type[node] = NODE_TYPE_INDEX.get(key, -1)
        qubits[node, : len(qs)] = qs
        for k, q in enumerate(qs[:2]):
            coh[node, 2 * k : 2 * k + 2] = lookup(q)
        preds = []
        for q in qs:
            p = last[q]
            if p not in preds:
                preds.append(p)
            last[q] = node
        src.extend(preds)
        dst.extend([node] * len(preds))
        node += 1

    edge_index = np.asarray([src, dst], dtype=np.int64).reshape(2, -1)
    return CircuitGraph(node_type, qubits, coh, position, edge_index)


def initial_layout_indices(compiled: QuantumCircuit) -> list[int] | None:
    """Logical-qubit -> physical-qubit map of a transpiled circuit (None if no layout)."""
    layout = getattr(compiled, "layout", None)
    if layout is None:
        return None
    try:
        return list(layout.initial_index_layout(filter_ancillas=True))
    except Exception:
        return None
