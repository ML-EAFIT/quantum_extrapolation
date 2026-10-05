"""Unit tests: python -m pytest tests -q"""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qetime.active_learning import greedy_sampling_x, sample_size
from qetime.circuits import family_of
from qetime.dataset import FeatureScaler, Sample, node_column_mask, to_pyg
from qetime.features import (
    GLOBAL_FEATURE_NAMES,
    NODE_TYPE_INDEX,
    circuit_graph,
    global_features,
    node_dim,
    supermarq_features,
)
from qetime.measure import required_repeats
from qetime.metrics import all_metrics


def bell():
    """The circuit of the paper's Fig. 1."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_node_dim_is_178():
    assert node_dim(127) == 178


def test_supermarq_bell():
    f = supermarq_features(bell())
    assert f["program_communication"] == pytest.approx(1.0)  # both qubits talk to each other
    assert f["critical_depth"] == pytest.approx(1.0)  # the only CX is on the critical path
    assert f["entanglement_ratio"] == pytest.approx(0.5)  # 1 two-qubit gate / 2 gates
    assert f["parallelism"] == pytest.approx(0.0)  # (2/2 - 1)/(2 - 1)
    assert f["liveness"] == pytest.approx(0.75)  # 3 active qubit-steps / (2 qubits x depth 2)


def test_global_feature_vector():
    v = dict(zip(GLOBAL_FEATURE_NAMES, global_features(bell())))
    assert len(GLOBAL_FEATURE_NAMES) == 51
    assert v["h"] == 1 and v["cx"] == 1 and v["num_qubits"] == 2
    assert v["depth"] == 3  # h, cx, measure


def test_graph_matches_fig4():
    coh = np.array([[100.0, 50.0], [200.0, 80.0]])
    g = circuit_graph(bell(), coh)
    # q0_in, q1_in, H, CX, M0, M1
    assert g.num_nodes == 6
    assert sorted(map(tuple, g.edge_index.T.tolist())) == [(0, 2), (1, 3), (2, 3), (3, 4), (3, 5)]
    assert g.node_type.tolist() == [NODE_TYPE_INDEX[t] for t in ["qubit_in", "qubit_in", "h", "cx", "measure", "measure"]]
    x = g.to_dense(127)
    assert x.shape == (6, 178)
    cx = x[3]
    assert cx[NODE_TYPE_INDEX["cx"]] == 1
    assert cx[46] == 1 and cx[47] == 1 and cx[46:173].sum() == 2  # qubits 0 and 1
    assert cx[173:177].tolist() == [100.0, 50.0, 200.0, 80.0]  # T1/T2 of both qubits
    assert cx[177] == 3  # sequential position


def test_idle_wires_dropped():
    qc = QuantumCircuit(5)
    qc.h(3)
    g = circuit_graph(qc)
    assert g.num_nodes == 2 and g.num_edges == 1


def test_family_names():
    assert family_of("qwalk-noancilla_indep_qiskit_9") == "qwalk-noancilla"
    assert family_of("ghz_indep_qiskit_10.qasm") == "ghz"


def test_required_repeats_eq1():
    # identical measurements need no more than the observations taken
    assert required_repeats([1.0, 1.0, 1.0]) == 0
    t = [1.0, 1.2, 0.8]
    expected = math.ceil((100 * 1.96 * np.std(t, ddof=1) / (25 * np.mean(t))) ** 2)
    assert required_repeats(t) == expected


def test_sample_size_matches_paper():
    # 3020 (circuit, device) pairs at 95 % confidence / 5 % margin -> ~340 samples
    assert sample_size(3020) in (340, 341)


def test_gsx_starts_at_centroid_and_spreads():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(200, 3))
    H = rng.normal(size=(200, 4))
    sel = greedy_sampling_x(G, H, 10)
    assert len(set(sel)) == 10
    Gn = (G - G.min(0)) / (G.max(0) - G.min(0))
    Hn = (H - H.min(0)) / (H.max(0) - H.min(0))
    d = 0.5 * np.linalg.norm(Gn - Gn.mean(0), axis=1) + 0.5 * np.linalg.norm(Hn - Hn.mean(0), axis=1)
    assert sel[0] == int(np.argmin(d))


def test_metrics():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = all_metrics(y, y)
    assert m["mse"] == 0 and m["r2"] == 1 and m["nmse"] == 0
    m = all_metrics(y, np.full(4, y.mean()))
    assert m["r2"] == pytest.approx(0) and m["nmse"] == pytest.approx(1)
    p = y + np.array([0.3, -0.2, 0.1, 0.4])
    m = all_metrics(y, p)
    assert m["r2"] == pytest.approx(1 - m["nmse"])


def _toy_samples(n=24):
    rng = np.random.default_rng(1)
    out = []
    for i in range(n):
        q = int(rng.integers(2, 6))
        qc = QuantumCircuit(q)
        for _ in range(int(rng.integers(1, 15))):
            a, b = rng.choice(q, 2, replace=False)
            qc.h(int(a))
            qc.cx(int(a), int(b))
        qc.measure_all()
        g = circuit_graph(qc, np.full((127, 2), 100.0))
        out.append(Sample(f"toy_{i}", "sim", "toy", global_features(qc), g, y=float(len(qc.data) * 0.1 + 1)))
    return out


def test_model_forward_backward_and_ablation_masks():
    import torch
    from torch_geometric.loader import DataLoader

    from qetime.model import ExecTimeModel

    samples = _toy_samples()
    scaler = FeatureScaler().fit(samples)
    data = to_pyg(samples, scaler)
    batch = next(iter(DataLoader(data, batch_size=8)))
    for drop, dim in [((), 178), (("node_type",), 132), (("t1t2", "node_index"), 173)]:
        mask = node_column_mask(127, drop)
        model = ExecTimeModel(global_dim=scaler.global_dim, node_mask=mask)
        assert model.nodes.out_dim == dim
        out = model(batch)
        assert out.shape == (8,)
        torch.nn.functional.mse_loss(out, batch.y.view(-1)).backward()
    # the in-model expansion equals the dense features of the scaler
    model = ExecTimeModel(global_dim=scaler.global_dim)
    dense = torch.from_numpy(scaler.transform_graph(samples[0].graph))
    assert torch.allclose(model.nodes(data[0]), dense, atol=1e-6)


def test_training_learns_toy_problem():
    from qetime.train import TrainConfig, run_cv, run_split

    samples = _toy_samples(40)
    cfg = TrainConfig(epochs=60, batch_size=8, hidden=32, verbose=False)
    res = run_split(samples, cfg)
    assert res["test"]["r2"] > 0.5
    cv = run_cv(samples, TrainConfig(epochs=5, batch_size=8, hidden=16, verbose=False), k=3)
    assert len(cv["per_fold"]) == 3 and len(cv["predictions"]) == 40


def test_scaler_handles_gates_unseen_in_training():
    samples = _toy_samples(10)
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.measure_all()
    unseen = Sample("ccx", "sim", "other", global_features(qc), circuit_graph(qc), y=1.0)
    scaler = FeatureScaler().fit(samples, global_cols_from=samples + [unseen])
    assert "ccx" in scaler.global_names
    assert np.abs(scaler.transform_global(unseen.global_raw)).max() < 100
