"""Analyses: RQ1 statistics, IBM estimate evaluation, SHAP importance of global features."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .dataset import Sample, to_pyg
from .metrics import all_metrics

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- RQ1
def describe(times: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics of execution times per backend (Tables 1 and 6)."""
    return times.groupby("backend")["time_s"].describe().T


def compare_backends(times: pd.DataFrame, a: str, b: str) -> dict:
    """Shapiro-Wilk per backend, paired Wilcoxon signed-rank and Spearman rho (Tables 2 and 7)."""
    wide = times.pivot_table(index="circuit", columns="backend", values="time_s").dropna(subset=[a, b])
    x, y = wide[a].values, wide[b].values
    sw_a, sw_b = stats.shapiro(x), stats.shapiro(y)
    wil = stats.wilcoxon(x, y)
    sp = stats.spearmanr(x, y)
    return {
        "pair": f"{a} vs {b}",
        "n_common": int(len(x)),
        f"shapiro_p_{a}": float(sw_a.pvalue),
        f"shapiro_p_{b}": float(sw_b.pvalue),
        "wilcoxon_p": float(wil.pvalue),
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
    }


def max_ratio(times: pd.DataFrame) -> pd.Series:
    """Longest / shortest execution time per backend (e.g. 271x on FakeSherbrooke)."""
    g = times.groupby("backend")["time_s"]
    return g.max() / g.min()


def feature_associations(samples: Sequence[Sample]) -> pd.DataFrame:
    """Spearman rho and Pearson r of simple target-independent features vs. time (Tables 5 and 8)."""
    rows = []
    feats = ["depth", "num_qubits", "two_qubit_gates", "num_gates"]
    by_backend: dict[str, list[Sample]] = {}
    for s in samples:
        by_backend.setdefault(s.backend, []).append(s)
    for backend, ss in sorted(by_backend.items()):
        y = np.array([s.y for s in ss])
        for f in feats:
            x = np.array([s.stats[f] for s in ss])
            rows.append({
                "backend": backend, "feature": f,
                "spearman_rho": float(stats.spearmanr(x, y).statistic),
                "pearson_r": float(stats.pearsonr(x, y).statistic),
            })
    return pd.DataFrame(rows).pivot(index="feature", columns="backend", values=["spearman_rho", "pearson_r"]).loc[feats]


def cross_spearman(times: pd.DataFrame, group_a: Sequence[str], group_b: Sequence[str]) -> pd.DataFrame:
    """Spearman rho between each pair of backends (e.g. simulators vs devices, Table 15)."""
    wide = times.pivot_table(index="circuit", columns="backend", values="time_s")
    rows = []
    for a, b in itertools.product(group_a, group_b):
        sub = wide[[a, b]].dropna()
        if len(sub) < 3:
            continue
        r = stats.spearmanr(sub[a], sub[b])
        rows.append({"a": a, "b": b, "n": len(sub), "rho": float(r.statistic), "p": float(r.pvalue)})
    return pd.DataFrame(rows)


def ibm_estimate_metrics(hw_csv: str | Path) -> dict:
    """Accuracy of IBM's pre-execution estimate against the measured quantum time (Table 9)."""
    df = pd.read_csv(hw_csv)
    df = df[(df["status"] == "ok")].dropna(subset=["estimated_s", "actual_s"])
    per = df.groupby(["circuit", "backend"])[["estimated_s", "actual_s"]].mean()
    return all_metrics(per["actual_s"].values, per["estimated_s"].values)


def plot_histograms(times: pd.DataFrame, out_png: str | Path, log_scale: bool = True) -> Path:
    """Histograms of execution times per backend (Figs. 7 and 8)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for backend, g in times.groupby("backend"):
        v = np.log(g.time_s.values) if log_scale else g.time_s.values
        _, _, patches = ax.hist(v, bins=60, alpha=0.5, label=backend)
        color = patches[0].get_facecolor()[:3]
        ax.axvline(np.mean(v), ls="--", lw=1, color=color)
        ax.axvline(np.median(v), ls=":", lw=1, color=color)
    ax.set_xlabel("log(execution time [s])" if log_scale else "execution time [s]")
    ax.set_ylabel("frequency (dashed: mean, dotted: median)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return Path(out_png)


# --------------------------------------------------------------------------- SHAP (RQ2/RQ3)
def global_shap(ckpt, background: Sequence[Sample], explain: Sequence[Sample], max_background: int = 200) -> pd.DataFrame:
    """Mean |SHAP| of each global feature.

    The graph branch is collapsed to its pooled embedding, so the explained function is
    the model head over [graph embedding, global features]; SHAP values are estimated
    with ``shap.GradientExplainer`` (expected gradients) and reported in seconds when
    the model's target is standardized seconds.
    """
    import shap
    import torch

    from .train import node_budget_batches

    model = ckpt.build_model().eval()
    scaler = ckpt.get_scaler()

    def embed(samples):
        data = to_pyg(samples, scaler, model.use_graph)
        parts = []
        with torch.no_grad():
            for batch in node_budget_batches(data, 128, 100_000):
                z = []
                if model.use_graph:
                    z.append(model.graph_embedding(batch))
                z.append(batch.g)
                parts.append(torch.cat(z, dim=1))
        return torch.cat(parts)

    rng = np.random.default_rng(0)
    bg = list(background)
    if len(bg) > max_background:
        bg = [bg[i] for i in rng.choice(len(bg), max_background, replace=False)]
    Zb, Ze = embed(bg), embed(explain)
    h = model.kwargs["hidden"] if model.use_graph else 0

    class Head(torch.nn.Module):
        def forward(self, z):
            return model.head_from(z[:, :h] if h else None, z[:, h:]).unsqueeze(-1)

    head = Head()
    explainer = shap.GradientExplainer(head, Zb)
    values = explainer.shap_values(Ze)
    values = np.asarray(values[0] if isinstance(values, list) else values)
    values = values.reshape(values.shape[0], values.shape[1])
    scale = scaler.y_std if scaler.target == "standard" else 1.0
    imp = np.abs(values[:, h:]).mean(axis=0) * scale
    return (
        pd.DataFrame({"feature": scaler.global_names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
