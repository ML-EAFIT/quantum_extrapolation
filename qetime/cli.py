"""Command-line interface: ``python -m qetime <command> ...`` (or ``qetime`` once installed)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import warnings
from dataclasses import fields
from pathlib import Path

PAPER_REPO = "https://github.com/mooselab/Quantum-Execution-Time-Prediction"
log = logging.getLogger("qetime")


# --------------------------------------------------------------------------- utilities
def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("qiskit", "stevedore", "qiskit_ibm_runtime", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def _circuit_files(args):
    from .circuits import iter_circuit_files

    names = None
    if getattr(args, "selection", None):
        import pandas as pd

        names = set(pd.read_csv(args.selection)["circuit"])
    return list(iter_circuit_files(args.circuits, args.min_qubits, args.max_qubits, names))


def _train_config(args):
    from .train import TrainConfig

    cfg = TrainConfig()
    for f in fields(TrainConfig):
        if f.name == "verbose":
            continue
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, tuple(v) if f.name == "drop_node_groups" else v)
    if getattr(args, "graph_only", False):
        cfg.use_global = False
    if getattr(args, "global_only", False):
        cfg.use_graph = False
    return cfg


def _load_data(path, backends=None, max_nodes=None):
    from .dataset import filter_samples, load_samples

    samples, meta = load_samples(path)
    samples = filter_samples(samples, backends)
    if max_nodes:
        before = len(samples)
        samples = [s for s in samples if s.graph.num_nodes <= max_nodes]
        log.warning("--max-nodes %d: kept %d of %d samples", max_nodes, len(samples), before)
    if not samples:
        sys.exit("no labelled samples left")
    log.info("loaded %d samples from %s (%s)", len(samples), path, meta)
    return samples, meta


# --------------------------------------------------------------------------- commands
def cmd_fetch_paper_data(args):
    """Download the authors' replication package (1,510 circuits + measured times)."""
    dest = Path(args.dest)
    if (dest / "data" / "quantum_circuits").exists():
        log.info("already present: %s", dest)
        return
    tmp = dest.with_name(dest.name + "_clone")
    try:
        subprocess.run(["git", "clone", "--depth", "1", PAPER_REPO, str(tmp)], check=True)
    except Exception:
        import io
        import urllib.request
        import zipfile

        log.info("git unavailable, downloading zip")
        with urllib.request.urlopen(PAPER_REPO + "/archive/refs/heads/main.zip") as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(tmp.parent)
        (tmp.parent / "Quantum-Execution-Time-Prediction-main").rename(tmp)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp / "data"), str(dest / "data"))
    shutil.rmtree(tmp, ignore_errors=True)
    n = len(list((dest / "data" / "quantum_circuits").glob("*.qasm")))
    log.info("paper data in %s: %d circuits, times: %s", dest / "data", n,
             ", ".join(p.name for p in sorted((dest / "data").glob("*.csv"))))


def cmd_generate_circuits(args):
    from .circuits import generate_mqt_circuits

    files = generate_mqt_circuits(args.out, args.benchmarks, args.min_qubits, args.max_qubits, args.step)
    log.info("%d circuits in %s", len(files), args.out)


def cmd_measure_sim(args):
    from .measure import measure_simulators

    files = _circuit_files(args)
    log.info("measuring %d circuits on %s", len(files), ", ".join(args.backends))
    measure_simulators(
        files, args.backends, args.out, compiled_dir=args.compiled_dir, shots=args.shots,
        min_repeats=args.repeats, max_repeats=max(args.repeats, args.max_repeats or args.repeats),
        timeout=args.timeout, opt_level=args.opt_level, seed=args.seed, precision=args.precision,
    )


def cmd_build_dataset(args):
    from .dataset import build_samples, load_times, save_samples

    times = load_times(args.times) if args.times else None
    files = _circuit_files(args)
    samples = build_samples(files, args.backends, args.graph_source, args.compiled_dir, args.opt_level,
                            args.seed, args.workers, times)
    meta = {"graph_source": args.graph_source, "backends": args.backends, "opt_level": args.opt_level}
    save_samples(samples, args.out, meta)
    labelled = sum(s.y == s.y for s in samples)
    log.info("wrote %s: %d samples (%d with measured times)", args.out, len(samples), labelled)


def cmd_select(args):
    import pandas as pd

    from .active_learning import sample_size, select
    from .dataset import build_samples, load_samples

    if args.pool:
        pool, _ = load_samples(args.pool)
    else:
        pool = build_samples(_circuit_files(args), args.backends, args.graph_source, args.compiled_dir,
                             workers=args.workers)
    k = args.k or sample_size(len(pool), args.confidence, args.margin)
    idx = select(pool, k, args.alpha)
    df = pd.DataFrame([{"order": i, "circuit": pool[j].circuit, "backend": pool[j].backend,
                        "family": pool[j].family} for i, j in enumerate(idx)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    log.info("selected %d of %d (circuit, backend) pairs -> %s\n%s", len(df), len(pool), args.out,
             df.backend.value_counts().to_string())


def cmd_hw_submit(args):
    import pandas as pd

    from .backends import canonical_name
    from .hardware import submit

    files = _circuit_files(args)
    if args.selection:
        sel = pd.read_csv(args.selection)
        want = set(sel.loc[sel.backend == canonical_name(args.backend), "circuit"])
        files = [f for f in files if f.stem in want]
    submit(files, args.backend, args.out, args.shots, args.repeats, args.opt_level, args.seed)


def cmd_hw_collect(args):
    from .hardware import collect

    collect(args.out)


def _save_run(out: Path, result: dict, ckpt=None, meta=None):
    out.mkdir(parents=True, exist_ok=True)
    if ckpt is not None:
        ckpt.info.update(meta or {})
        ckpt.save(out / "model.pt")
    result["predictions"].to_csv(out / "predictions.csv", index=False)
    summary = {k: v for k, v in result.items() if k not in ("predictions", "checkpoint", "per_fold", "history")}
    if "per_fold" in result:
        result["per_fold"].to_csv(out / "per_fold.csv", index=False)
    if "history" in result:
        summary["best_epoch"] = result["history"]["best_epoch"]
        summary["train_seconds"] = result["history"]["seconds"]
    _write_json(summary, out / "metrics.json")
    return summary


def cmd_train(args):
    from .train import run_split

    samples, meta = _load_data(args.data, args.backends, args.max_nodes)
    cfg = _train_config(args)
    res = run_split(samples, cfg, ratios=tuple(args.ratios))
    summary = _save_run(Path(args.out), res, res["checkpoint"], {"graph_source": meta.get("graph_source", "logical")})
    log.info("test: %s", json.dumps(summary["test"]))


def cmd_cv(args):
    from .train import Checkpoint, family_folds, run_cv

    samples, meta = _load_data(args.data, args.backends, args.max_nodes)
    cfg = _train_config(args)
    pretrained = Checkpoint.load(args.pretrained) if args.pretrained else None
    folds = names = None
    if args.by_family:
        folds, names = family_folds(samples, args.min_family_size)
        log.info("algorithm-family folds: %s", {n: len(f) for n, f in zip(names, folds)})
    res = run_cv(samples, cfg, folds=folds, k=args.folds, fold_names=names, pretrained=pretrained)
    summary = _save_run(Path(args.out), res)
    log.info("mean over folds: %s | pooled: %s", json.dumps(summary["mean"]), json.dumps(summary["pooled"]))


def cmd_ablate(args):
    """Feature ablations of Tables 10-13: full / graph-only / global-only / drop each node group / per backend."""
    import copy

    import pandas as pd

    from .dataset import filter_samples
    from .train import fit_checkpoint, run_cv, run_split, split_three

    samples, meta = _load_data(args.data, args.backends, args.max_nodes)
    base = _train_config(args)
    variants = [("full", {})]
    variants += [("graph_only", {"use_global": False}), ("global_only", {"use_graph": False})]
    variants += [(f"without_{g}", {"drop_node_groups": (g,)}) for g in ("node_type", "qubit_index", "t1t2", "node_index")]
    if args.only:
        variants = [v for v in variants if v[0] in args.only]
    backends = sorted({s.backend for s in samples})
    rows = []
    pre_samples = None
    if args.pretrain_data:
        pre_samples, _ = _load_data(args.pretrain_data, None, args.max_nodes)
    runs = [(n, kw, samples) for n, kw in variants]
    if args.per_backend and len(backends) > 1 and not args.only:
        runs += [(f"{b}_only", {}, filter_samples(samples, [b])) for b in backends]
    for name, kw, data in runs:
        cfg = copy.deepcopy(base)
        for k, v in kw.items():
            setattr(cfg, k, v)
        log.info("=== variant %s (%d samples)", name, len(data))
        if args.mode == "split":
            res = run_split(data, cfg)
            rows.append({"variant": name, **{k: res["test"][k] for k in ("mse", "r2", "nmse", "n")}})
        else:
            pretrained = None
            if pre_samples is not None:
                tr, va, _ = split_three(len(pre_samples), (0.8, 0.1, 0.1), cfg.seed)
                pcfg = copy.deepcopy(cfg)
                pcfg.batch_size = args.pretrain_batch_size
                pcfg.epochs = args.pretrain_epochs or cfg.epochs
                pretrained, _ = fit_checkpoint([pre_samples[i] for i in tr], [pre_samples[i] for i in va], pcfg,
                                               pre_samples, tag=f"[pretrain {name}]")
            res = run_cv(data, cfg, k=args.folds, pretrained=pretrained)
            rows.append({"variant": name, **res["mean"], "n": len(data)})
        log.info("%s -> %s", name, rows[-1])
        pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_grid_search(args):
    from .train import grid_search

    samples, _ = _load_data(args.data, args.backends, args.max_nodes)
    df = grid_search(samples, _train_config(args), args.epochs_grid, args.batch_grid)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.head(10).to_string(index=False))


def cmd_analyze(args):
    import pandas as pd

    from .analysis import compare_backends, cross_spearman, describe, feature_associations, ibm_estimate_metrics, max_ratio, plot_histograms
    from .dataset import load_samples, load_times

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    times = load_times(args.times)
    report = {}
    desc = describe(times)
    desc.to_csv(out / "time_statistics.csv")
    print("Execution-time statistics (s):\n", desc.round(3).to_string())
    print("Longest/shortest ratio:\n", max_ratio(times).round(1).to_string())
    backends = sorted(times.backend.unique())
    pairs = [tuple(p) for p in args.pairs] if args.pairs else [(a, b) for i, a in enumerate(backends) for b in backends[i + 1:]]
    tests = [compare_backends(times, a, b) for a, b in pairs]
    report["backend_comparisons"] = tests
    for t in tests:
        print(json.dumps(t, indent=1))
    if args.data:
        samples = []
        for d in args.data:
            ss, _ = load_samples(d)
            samples += [s for s in ss if s.y == s.y]
        assoc = feature_associations(samples)
        assoc.to_csv(out / "feature_associations.csv")
        print("Feature associations:\n", assoc.round(3).to_string())
    if args.sim and args.hw:
        cs = cross_spearman(times, args.sim, args.hw)
        cs.to_csv(out / "sim_vs_hw_spearman.csv", index=False)
        print("Simulator vs device Spearman:\n", cs.to_string(index=False))
    if args.ibm_csv:
        report["ibm_estimate"] = ibm_estimate_metrics(args.ibm_csv)
        print("IBM estimate vs actual:", report["ibm_estimate"])
    try:
        plot_histograms(times, out / "histograms.png", log_scale=not args.linear)
    except ImportError:
        log.info("matplotlib not installed; skipping histograms")
    _write_json(report, out / "report.json")


def cmd_shap(args):
    from .analysis import global_shap
    from .train import Checkpoint, split_three

    ckpt = Checkpoint.load(args.model)
    samples, _ = _load_data(args.data, args.backends, args.max_nodes)
    tr, va, te = split_three(len(samples), (0.8, 0.1, 0.1), ckpt.config.get("seed", 1234))
    df = global_shap(ckpt, [samples[i] for i in tr], [samples[i] for i in (te if args.test_only else range(len(samples)))])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.head(args.top).to_string(index=False))


def cmd_predict(args):
    import numpy as np

    from .dataset import featurize, to_pyg
    from .train import Checkpoint, predict_seconds

    ckpt = Checkpoint.load(args.model)
    scaler = ckpt.get_scaler()
    source = ckpt.info.get("graph_source", "logical")
    samples = [featurize(c, args.backend, source, args.compiled_dir) for c in args.circuits]
    model = ckpt.build_model()
    pred = predict_seconds(model, to_pyg(samples, scaler, model.use_graph), scaler)
    for s, p in zip(samples, np.atleast_1d(pred)):
        print(f"{s.circuit}\t{s.backend}\t{p:.3f} s")


# --------------------------------------------------------------------------- parser
def _add_circuit_args(p, required=True):
    p.add_argument("--circuits", required=required, help="directory with .qasm/.qpy circuits")
    p.add_argument("--min-qubits", type=int, default=2)
    p.add_argument("--max-qubits", type=int, default=127)


def _add_train_args(p, batch_size=128):
    p.add_argument("--data", required=True, help="dataset .pkl from build-dataset")
    p.add_argument("--backends", nargs="*", help="restrict to these backends")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=178)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--graph-only", action="store_true")
    p.add_argument("--global-only", action="store_true")
    p.add_argument("--drop-node-groups", nargs="*", choices=["node_type", "qubit_index", "t1t2", "node_index"])
    p.add_argument("--target", choices=["standard", "log"], default="standard")
    p.add_argument("--no-log-counts", dest="log_counts", action="store_false",
                   help="z-score raw gate counts instead of log1p(counts)")
    p.add_argument("--patience", type=int, default=0, help="early stopping patience (0 = off)")
    p.add_argument("--max-nodes", type=int, help="drop graphs with more nodes (for limited hardware)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qetime", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch-paper-data", help="download the paper's circuits and measured times")
    p.add_argument("--dest", default="data/paper")
    p.set_defaults(func=cmd_fetch_paper_data)

    p = sub.add_parser("generate-circuits", help="generate MQT Bench target-independent circuits")
    p.add_argument("--out", required=True)
    p.add_argument("--benchmarks", nargs="*")
    p.add_argument("--min-qubits", type=int, default=2)
    p.add_argument("--max-qubits", type=int, default=127)
    p.add_argument("--step", type=int, default=1)
    p.set_defaults(func=cmd_generate_circuits)

    p = sub.add_parser("measure-sim", help="measure execution times on noisy simulators")
    _add_circuit_args(p)
    p.add_argument("--selection", help="CSV with a 'circuit' column to restrict to")
    p.add_argument("--backends", nargs="+", default=["sherbrooke", "washington"])
    p.add_argument("--out", required=True, help="CSV (appended to, resumable)")
    p.add_argument("--compiled-dir", help="cache directory for compiled circuits")
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-repeats", type=int, default=0, help="adaptive: run up to this many repeats if Eq. 1 asks for more")
    p.add_argument("--precision", type=float, default=25.0, help="r in Eq. 1 (percent)")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds per execution")
    p.add_argument("--opt-level", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.set_defaults(func=cmd_measure_sim)

    p = sub.add_parser("build-dataset", help="featurize circuits and attach measured times")
    _add_circuit_args(p)
    p.add_argument("--selection", help="CSV with a 'circuit' column to restrict to")
    p.add_argument("--times", nargs="*", help="time CSVs (omit to build an unlabelled pool)")
    p.add_argument("--backends", nargs="+", required=True)
    p.add_argument("--graph-source", choices=["logical", "compiled"], default="logical")
    p.add_argument("--compiled-dir", default="work/compiled")
    p.add_argument("--opt-level", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_build_dataset)

    p = sub.add_parser("select", help="active learning (GSx) selection of circuits for hardware")
    _add_circuit_args(p, required=False)
    p.add_argument("--pool", help="unlabelled dataset .pkl (instead of --circuits/--backends)")
    p.add_argument("--backends", nargs="+", default=["osaka", "kyoto"])
    p.add_argument("--graph-source", choices=["logical", "compiled"], default="logical")
    p.add_argument("--compiled-dir", default="work/compiled")
    p.add_argument("--workers", type=int)
    p.add_argument("--k", type=int, help="number of samples (default: sample size at --confidence/--margin)")
    p.add_argument("--confidence", type=float, default=0.95)
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("hw-submit", help="submit circuits to an IBM Quantum device")
    _add_circuit_args(p)
    p.add_argument("--selection", help="CSV from 'select' (rows for this backend are used)")
    p.add_argument("--backend", required=True, help="e.g. ibm_fez")
    p.add_argument("--out", required=True)
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--opt-level", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.set_defaults(func=cmd_hw_submit)

    p = sub.add_parser("hw-collect", help="fetch actual quantum seconds of finished jobs")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_hw_collect)

    p = sub.add_parser("train", help="train/evaluate on an 8:1:1 split and save the model (RQ2)")
    _add_train_args(p)
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    p.add_argument("--out", required=True, help="output directory")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("cv", help="k-fold or algorithm-family cross-validation (RQ3, Section 7)")
    _add_train_args(p, batch_size=32)
    p.add_argument("--pretrained", help="checkpoint to fine-tune (e.g. the simulator model)")
    p.add_argument("--keep-target-scale", dest="refit_target", action="store_false",
                   help="keep the pre-trained model's target standardization when fine-tuning")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--by-family", action="store_true")
    p.add_argument("--min-family-size", type=int, default=10)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_cv)

    p = sub.add_parser("ablate", help="feature ablations (Tables 10-13)")
    _add_train_args(p)
    p.add_argument("--mode", choices=["split", "cv"], default="split")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--pretrain-data", help="cv mode: simulator dataset used to pre-train each variant")
    p.add_argument("--pretrain-epochs", type=int)
    p.add_argument("--pretrain-batch-size", type=int, default=128)
    p.add_argument("--per-backend", action="store_true", help="also train per-backend models")
    p.add_argument("--only", nargs="*", help="run only these variants")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("grid-search", help="epochs x batch-size grid search")
    _add_train_args(p)
    p.add_argument("--epochs-grid", nargs="*", type=int, default=list(range(100, 801, 100)))
    p.add_argument("--batch-grid", nargs="*", type=int, default=list(range(32, 161, 32)))
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_grid_search)

    p = sub.add_parser("analyze", help="RQ1 statistics and IBM-estimate evaluation")
    p.add_argument("--times", nargs="+", required=True)
    p.add_argument("--data", nargs="*", help="datasets for feature-time associations")
    p.add_argument("--pairs", nargs=2, action="append", metavar=("A", "B"))
    p.add_argument("--sim", nargs="*", help="simulator backends for the sim-vs-device table")
    p.add_argument("--hw", nargs="*", help="device backends for the sim-vs-device table")
    p.add_argument("--ibm-csv", help="hw-submit/hw-collect CSV with IBM estimates")
    p.add_argument("--linear", action="store_true", help="linear instead of log histograms")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("shap", help="SHAP importance of the global features")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--backends", nargs="*")
    p.add_argument("--max-nodes", type=int)
    p.add_argument("--test-only", action="store_true", help="explain only the test split")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_shap)

    p = sub.add_parser("predict", help="predict execution time of circuits on a backend")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--compiled-dir")
    p.add_argument("circuits", nargs="+")
    p.set_defaults(func=cmd_predict)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
