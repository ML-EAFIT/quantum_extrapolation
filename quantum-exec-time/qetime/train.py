"""Training and evaluation (paper Sections 5.2 and 5.4, Section 7).

* :func:`run_split`     - fixed 8:1:1 train/validation/test split (simulators, RQ2);
                          the checkpoint with the lowest validation MSE is kept.
* :func:`run_cv`        - k-fold cross-validation (real devices, RQ3). In every fold the
                          remaining data is split 8:1 into training and validation
                          subsets; the epoch with the lowest validation MSE is evaluated
                          on the held-out fold. Optionally fine-tunes a pre-trained
                          (simulator) checkpoint.
* :func:`family_folds`  - leave-one-algorithm-family-out folds (Section 7).
* :func:`grid_search`   - epochs x batch-size grid search on the validation split.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .dataset import FeatureScaler, Sample, node_column_mask, to_pyg
from .metrics import all_metrics
from .model import ExecTimeModel

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-4
    hidden: int = 178
    num_layers: int = 3
    heads: int = 1
    dropout: float = 0.0
    use_graph: bool = True
    use_global: bool = True
    drop_node_groups: tuple[str, ...] = ()
    max_qubits: int = 127
    log_counts: bool = True
    target: str = "standard"
    seed: int = 1234
    device: str = "cpu"
    patience: int = 0  # 0 disables early stopping
    refit_target: bool = True  # when fine-tuning, re-standardize the target on the new data
    threads: int = 0
    verbose: bool = True


@dataclass
class Checkpoint:
    model_state: dict
    model_kwargs: dict
    scaler: dict
    node_mask: np.ndarray
    config: dict
    info: dict = field(default_factory=dict)

    def build_model(self) -> ExecTimeModel:
        model = ExecTimeModel(**self.model_kwargs)
        model.load_state_dict(self.model_state)
        return model

    def get_scaler(self) -> FeatureScaler:
        return FeatureScaler.from_state_dict(self.scaler)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> "Checkpoint":
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


# --------------------------------------------------------------------------- helpers
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _setup(cfg: TrainConfig) -> torch.device:
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    return torch.device(cfg.device)


def build_model(cfg: TrainConfig, scaler: FeatureScaler, node_mask: np.ndarray) -> ExecTimeModel:
    return ExecTimeModel(
        global_dim=scaler.global_dim,
        max_qubits=scaler.max_qubits,
        node_mask=node_mask,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        heads=cfg.heads,
        use_graph=cfg.use_graph,
        use_global=cfg.use_global,
        dropout=cfg.dropout,
    )


def node_budget_batches(data: list, batch_size: int, max_nodes: int):
    """Yield batches of at most ``batch_size`` graphs and (unless a single graph is larger) ``max_nodes`` nodes."""
    from torch_geometric.data import Batch

    cur, n = [], 0
    for d in data:
        if cur and (len(cur) >= batch_size or n + d.num_nodes > max_nodes):
            yield Batch.from_data_list(cur)
            cur, n = [], 0
        cur.append(d)
        n += d.num_nodes
    if cur:
        yield Batch.from_data_list(cur)


@torch.no_grad()
def predict_scaled(model: ExecTimeModel, data: list, batch_size: int = 128, device="cpu",
                   max_nodes: int = 100_000) -> np.ndarray:
    model.eval()
    out = []
    for batch in node_budget_batches(data, batch_size, max_nodes):
        out.append(model(batch.to(device)).cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


def predict_seconds(model, data, scaler: FeatureScaler, batch_size: int = 128, device="cpu") -> np.ndarray:
    return scaler.inverse_y(predict_scaled(model, data, batch_size, device))


def train_model(
    model: ExecTimeModel,
    train_data: list,
    val_data: list | None,
    val_y: np.ndarray | None,
    scaler: FeatureScaler,
    cfg: TrainConfig,
    tag: str = "",
) -> tuple[dict, dict]:
    """Train with MSE loss; return the state dict with the lowest validation MSE (seconds)."""
    device = _setup(cfg)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    history = {"train_loss": [], "val_mse": []}
    best_mse, best_state, best_epoch, since_best = math.inf, None, -1, 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = F.mse_loss(model(batch), batch.y.view(-1))
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_graphs
            count += batch.num_graphs
        history["train_loss"].append(total / max(count, 1))
        if val_data:
            pred = predict_seconds(model, val_data, scaler, device=device)
            vm = float(np.mean((pred - val_y) ** 2))
            history["val_mse"].append(vm)
            if vm < best_mse:
                best_mse, best_epoch, since_best = vm, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                since_best += 1
            if cfg.patience and since_best >= cfg.patience:
                break
        if cfg.verbose and (epoch % 25 == 0 or epoch == cfg.epochs - 1):
            msg = f"{tag} epoch {epoch + 1}/{cfg.epochs} loss={history['train_loss'][-1]:.4f}"
            if val_data:
                msg += f" val_mse={history['val_mse'][-1]:.4f} best={best_mse:.4f}@{best_epoch + 1}"
            log.info("%s (%.0fs)", msg, time.time() - t0)
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = cfg.epochs - 1
    history["best_epoch"] = best_epoch + 1
    history["best_val_mse"] = best_mse
    history["seconds"] = time.time() - t0
    return best_state, history


def _shuffle(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n)


def split_three(n: int, ratios=(0.8, 0.1, 0.1), seed: int = 1234) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perm = _shuffle(n, seed)
    r = np.asarray(ratios, float) / sum(ratios)
    n_train = int(round(n * r[0]))
    n_val = int(round(n * r[1]))
    return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]


def _prepare(samples, scaler, with_graph=True):
    return to_pyg(samples, scaler, with_graph), np.array([s.y for s in samples], dtype=float)


def _predictions_frame(samples: Sequence[Sample], pred: np.ndarray, **extra) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "circuit": [s.circuit for s in samples],
            "backend": [s.backend for s in samples],
            "family": [s.family for s in samples],
            "y": [s.y for s in samples],
            "pred": pred,
        }
    )
    for k, v in extra.items():
        df[k] = v
    return df


def _new_scaler(cfg: TrainConfig, train: Sequence[Sample], all_samples: Sequence[Sample]) -> FeatureScaler:
    return FeatureScaler(cfg.max_qubits, cfg.log_counts, cfg.target).fit(train, global_cols_from=all_samples)


# --------------------------------------------------------------------------- experiments
def fit_checkpoint(
    train: Sequence[Sample],
    val: Sequence[Sample] | None,
    cfg: TrainConfig,
    all_samples: Sequence[Sample] | None = None,
    pretrained: Checkpoint | None = None,
    tag: str = "",
) -> tuple[Checkpoint, dict]:
    """Train one model (optionally starting from ``pretrained``) and package it as a checkpoint."""
    set_seed(cfg.seed)
    if pretrained is not None:
        for key in ("use_graph", "use_global", "hidden", "num_layers"):
            if pretrained.model_kwargs.get(key) != getattr(cfg, key):
                log.warning("pretrained model has %s=%s; ignoring the requested %s", key,
                            pretrained.model_kwargs.get(key), getattr(cfg, key))
        scaler = pretrained.get_scaler()
        if cfg.refit_target:
            # inputs must keep the pre-trained scaling; the target range of the new
            # backend (e.g. ~8 s on devices vs ~1 s on simulators) is re-standardized
            y = scaler._y_forward(np.array([s.y for s in train], dtype=np.float64))
            scaler.y_mean, scaler.y_std = float(y.mean()), float(y.std() + 1e-8)
        node_mask = pretrained.node_mask
        model = pretrained.build_model()
    else:
        scaler = _new_scaler(cfg, train, all_samples or train)
        node_mask = node_column_mask(cfg.max_qubits, cfg.drop_node_groups)
        model = build_model(cfg, scaler, node_mask)
    train_data, _ = _prepare(train, scaler, model.use_graph)
    val_data, val_y = _prepare(val, scaler, model.use_graph) if val else (None, None)
    state, hist = train_model(model, train_data, val_data, val_y, scaler, cfg, tag)
    ckpt = Checkpoint(
        model_state=state,
        model_kwargs=model.kwargs,
        scaler=scaler.state_dict(),
        node_mask=node_mask,
        config=asdict(cfg),
        info={"history": hist, "fine_tuned": pretrained is not None},
    )
    return ckpt, hist


def evaluate_checkpoint(ckpt: Checkpoint, samples: Sequence[Sample], batch_size: int = 128) -> tuple[dict, np.ndarray]:
    scaler = ckpt.get_scaler()
    data, y = _prepare(samples, scaler, ckpt.model_kwargs.get("use_graph", True))
    device = torch.device(ckpt.config.get("device", "cpu"))
    model = ckpt.build_model().to(device)
    pred = predict_seconds(model, data, scaler, batch_size, device)
    return all_metrics(y, pred), pred


def run_split(
    samples: Sequence[Sample],
    cfg: TrainConfig,
    ratios=(0.8, 0.1, 0.1),
    pretrained: Checkpoint | None = None,
) -> dict:
    """Train on 80 %, select the epoch on 10 % (validation), report the 10 % test split."""
    tr, va, te = split_three(len(samples), ratios, cfg.seed)
    pick = lambda idx: [samples[i] for i in idx]  # noqa: E731
    ckpt, hist = fit_checkpoint(pick(tr), pick(va), cfg, samples, pretrained, tag="[split]")
    test_metrics, pred = evaluate_checkpoint(ckpt, pick(te))
    val_metrics, _ = evaluate_checkpoint(ckpt, pick(va))
    ckpt.info.update(test_metrics=test_metrics, val_metrics=val_metrics, n_train=len(tr))
    return {
        "test": test_metrics,
        "val": val_metrics,
        "checkpoint": ckpt,
        "history": hist,
        "predictions": _predictions_frame(pick(te), pred, split="test"),
    }


def kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    return [np.sort(f) for f in np.array_split(_shuffle(n, seed), k)]


def family_folds(samples: Sequence[Sample], min_size: int = 10) -> tuple[list[np.ndarray], list[str]]:
    """Leave-one-family-out folds; families smaller than ``min_size`` are merged with the next smallest group."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(s.family, []).append(i)
    items = sorted(([name], idx) for name, idx in groups.items())
    items = [(names, idx) for names, idx in items]
    while len(items) > 2:
        items.sort(key=lambda t: len(t[1]))
        if len(items[0][1]) >= min_size:
            break
        (n0, i0), (n1, i1) = items[0], items[1]
        items = [(n0 + n1, i0 + i1)] + items[2:]
    items.sort(key=lambda t: t[0])
    return [np.asarray(sorted(idx)) for _, idx in items], ["+".join(n) for n, _ in items]


def run_cv(
    samples: Sequence[Sample],
    cfg: TrainConfig,
    folds: list[np.ndarray] | None = None,
    k: int = 10,
    fold_names: list[str] | None = None,
    pretrained: Checkpoint | None = None,
    val_fraction: float = 1 / 9,
) -> dict:
    """Cross-validation; returns per-fold and averaged metrics plus out-of-fold predictions."""
    n = len(samples)
    folds = folds if folds is not None else kfold_indices(n, k, cfg.seed)
    fold_names = fold_names or [str(i) for i in range(len(folds))]
    rows, preds = [], []
    for fi, test_idx in enumerate(folds):
        rest = np.setdiff1d(np.arange(n), test_idx)
        rest = rest[_shuffle(len(rest), cfg.seed + fi)]
        n_val = max(1, int(round(len(rest) * val_fraction)))
        val_idx, train_idx = rest[:n_val], rest[n_val:]
        pick = lambda idx: [samples[i] for i in idx]  # noqa: E731
        ckpt, hist = fit_checkpoint(pick(train_idx), pick(val_idx), cfg, samples, pretrained, tag=f"[fold {fi + 1}/{len(folds)}]")
        m, pred = evaluate_checkpoint(ckpt, pick(test_idx))
        m.update(fold=fold_names[fi], best_epoch=hist["best_epoch"])
        rows.append(m)
        preds.append(_predictions_frame(pick(test_idx), pred, fold=fold_names[fi]))
        log.info("fold %s: R2=%.3f MSE=%.3f NMSE=%.3f (n=%d)", fold_names[fi], m["r2"], m["mse"], m["nmse"], m["n"])
    per_fold = pd.DataFrame(rows)
    oof = pd.concat(preds, ignore_index=True)
    return {
        "per_fold": per_fold,
        "mean": per_fold[["mse", "r2", "nmse", "mape", "spearman"]].mean().to_dict(),
        "std": per_fold[["mse", "r2", "nmse", "mape", "spearman"]].std().to_dict(),
        "pooled": all_metrics(oof.y.values, oof.pred.values),
        "predictions": oof,
    }


def grid_search(
    samples: Sequence[Sample],
    cfg: TrainConfig,
    epochs_grid: Sequence[int] = tuple(range(100, 801, 100)),
    batch_grid: Sequence[int] = tuple(range(32, 161, 32)),
    ratios=(0.8, 0.1, 0.1),
) -> pd.DataFrame:
    """Retrain from scratch for every (epochs, batch size) pair; score on the validation split."""
    tr, va, te = split_three(len(samples), ratios, cfg.seed)
    pick = lambda idx: [samples[i] for i in idx]  # noqa: E731
    rows = []
    for e in epochs_grid:
        for b in batch_grid:
            c = copy.deepcopy(cfg)
            c.epochs, c.batch_size = int(e), int(b)
            ckpt, _ = fit_checkpoint(pick(tr), None, c, samples, tag=f"[grid e={e} b={b}]")
            vm, _ = evaluate_checkpoint(ckpt, pick(va))
            tm, _ = evaluate_checkpoint(ckpt, pick(te))
            rows.append({"epochs": e, "batch_size": b, "val_mse": vm["mse"], "val_r2": vm["r2"],
                         "test_mse": tm["mse"], "test_r2": tm["r2"]})
            log.info("grid epochs=%d batch=%d val_mse=%.4f val_r2=%.3f", e, b, vm["mse"], vm["r2"])
    return pd.DataFrame(rows).sort_values("val_mse").reset_index(drop=True)
