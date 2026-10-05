# qetime — estimating the execution time of quantum circuits

An implementation of **Ma & Li, "Understanding and Estimating the Execution Time of
Quantum Circuits"** (ACM TOSEM 2025, [arXiv:2411.15631](https://arxiv.org/abs/2411.15631)):
a graph-transformer model that predicts how long a circuit runs on a simulator or an IBM
quantum computer, plus the data-collection, active-learning and analysis pipeline around it.

| Paper | Module | CLI |
|---|---|---|
| §4.1 circuits (MQT Bench, indep level) | `qetime/circuits.py` | `fetch-paper-data`, `generate-circuits` |
| §4.2 backends, T1/T2 | `qetime/backends.py` | — |
| §4.3–4.4 timing, Eq. 1 repeats, 10-min timeout | `qetime/measure.py` | `measure-sim` |
| §4.3, §6.1.3 IBM hardware + IBM's own estimate | `qetime/hardware.py` | `hw-submit`, `hw-collect` |
| §5.1 global (41-d) and graph (178-d/node) features | `qetime/features.py` | `build-dataset` |
| §5.2 model (Fig. 6) | `qetime/model.py` | — |
| §5.3 active learning, GSx (Alg. 1, Eq. 3–4) | `qetime/active_learning.py` | `select` |
| §5.4 split / 10-fold CV / fine-tuning, metrics | `qetime/train.py`, `qetime/metrics.py` | `train`, `cv`, `grid-search` |
| §6.1 RQ1 statistics (Tables 1–2, 5–9, 15) | `qetime/analysis.py` | `analyze` |
| §6.2–6.3 ablations & SHAP (Tables 10–13, Figs. 9–10) | `qetime/analysis.py`, `cli.py` | `ablate`, `shap` |
| §7 algorithm-family cross-validation (Table 14) | `qetime/train.py` | `cv --by-family` |

## Install

```bash
pip install -e ".[mqt,analysis,test]"     # Python >= 3.10
python -m pytest tests -q                  # 12 tests
```

Dependencies: Qiskit ≥ 1.2, qiskit-aer, qiskit-ibm-runtime, PyTorch, PyTorch Geometric
(pure-Python build is enough), numpy/scipy/pandas; optional `mqt.bench`, `shap`, `matplotlib`.

## Reproduce the paper on its own measurements

The authors published the 1,510 circuits and every measured execution time
(FakeSherbrooke, FakeWashington, ibm_osaka, ibm_kyoto). `fetch-paper-data` downloads them.

```bash
qetime fetch-paper-data --dest data/paper
P=data/paper/data

# RQ1: statistics of the measured times
qetime analyze --times $P/sherbrooke_time_taken.csv $P/washington_time_taken.csv \
                       $P/osaka_time_taken.csv $P/kyoto_time_taken.csv \
  --pairs sherbrooke washington --pairs osaka kyoto --sim sherbrooke washington --hw osaka kyoto \
  --out results/rq1

# datasets (features + times); osaka/kyoto use the bundled calibration snapshots
qetime build-dataset --circuits $P/quantum_circuits --backends sherbrooke washington \
  --times $P/sherbrooke_time_taken.csv $P/washington_time_taken.csv --out data/sim.pkl
qetime build-dataset --circuits $P/quantum_circuits --backends osaka kyoto \
  --times $P/osaka_time_taken.csv $P/kyoto_time_taken.csv --out data/hw.pkl

# RQ2: simulators, 8:1:1 split, 500 epochs, batch 128 -> results/sim/model.pt
qetime train --data data/sim.pkl --out results/sim --device cuda
qetime ablate --data data/sim.pkl --mode split --per-backend --out results/sim_ablation.csv --device cuda
qetime shap --model results/sim/model.pt --data data/sim.pkl --out results/sim_shap.csv

# RQ3: devices, 10-fold CV fine-tuning the simulator model (batch 32)
qetime cv --data data/hw.pkl --pretrained results/sim/model.pt --out results/hw_cv --device cuda
qetime cv --data data/hw.pkl --pretrained results/sim/model.pt --by-family --out results/hw_family_cv --device cuda
qetime ablate --data data/hw.pkl --mode cv --pretrain-data data/sim.pkl --batch-size 32 \
  --per-backend --out results/hw_ablation.csv --device cuda
```

## Collect your own data

```bash
qetime generate-circuits --out circuits/                       # MQT Bench (indep level), 2-127 qubits
qetime measure-sim --circuits circuits/ --backends sherbrooke washington \
  --out times/sim.csv --compiled-dir work/compiled             # 1024 shots, 3 repeats, 600 s timeout
qetime build-dataset --circuits circuits/ --backends sherbrooke washington --times times/sim.csv --out data/sim.pkl

# choose the circuits worth paying hardware time for (GSx; default k = 95 % / 5 % sample size)
qetime select --circuits circuits/ --backends ibm_fez ibm_torino --out selection.csv
export QISKIT_IBM_TOKEN=...                                    # or a saved Qiskit account
qetime hw-submit  --circuits circuits/ --selection selection.csv --backend ibm_fez --out times/hw.csv
qetime hw-collect --out times/hw.csv                           # re-run until all jobs are done
qetime analyze --times times/hw.csv --ibm-csv times/hw.csv --out results/ibm_estimate   # Table 9
```

`measure-sim` and `hw-submit` are resumable. `--max-repeats N` turns on adaptive
repetition: after the first `--repeats` runs, Eq. 1 decides whether more are needed.

## Predict

```bash
qetime predict --model results/sim/model.pt --backend sherbrooke my_circuit.qasm
```

## Implementation notes

**Graph source.** The paper's text describes graphs of the circuit *compiled* for each
backend; the authors' released code builds them from the target-independent circuit and
attaches the backend's T1/T2 by logical qubit index. Both are implemented:
`--graph-source logical` (default; what produced the published numbers) and
`--graph-source compiled`. Compiled graphs are 5–25× larger (e.g. 103,819 vs. 6,207
nodes for `qwalk-noancilla_9` on FakeSherbrooke).

**Model selection.** Following the paper's text, each run keeps the epoch with the lowest
MSE on a *validation* split and reports a separate test split/fold. The authors'
cross-validation script instead reports the best epoch measured on the held-out fold
itself, so expect somewhat lower numbers than the paper here.

**Scaling.** Scalers are fitted on the training split only. Count features are
`log1p`-transformed before z-scoring (`--no-log-counts` for plain z-scores as in the
authors' code); the target is z-scored seconds (`--target log` is available).
Global feature columns that are zero across the dataset are dropped — this gives exactly
the paper's 41 dimensions on its data.

**Memory.** Node features are stored compactly and expanded to 178-d vectors per batch
inside the model (numerically identical to dense input). The paper's simulator dataset
has 17 M nodes (~12 GB as dense float32); this way memory scales with the batch.

**Compute.** One epoch over the full simulator dataset takes ~6 min on two CPU cores;
use a GPU (`--device cuda`). `--max-nodes N` drops large graphs for quick experiments.
Activation memory grows with the number of nodes per batch: the device circuits average
~7.5 k nodes, so batch 32 needs roughly 7 GB; lower `--batch-size` if memory is short.

**Fine-tuning.** When a simulator checkpoint is fine-tuned on device data, the input
scaling of the checkpoint is kept and the target is re-standardized on the device
training data (device times are ~8 s vs ~1 s on simulators); `--keep-target-scale`
disables this. Global features that never vary in the training data (e.g. a gate that
only occurs in a held-out algorithm family) are fed as zero instead of extrapolated.

**Other choices.** GSx represents each graph by its mean node-feature vector (the
authors zero-pad node matrices to the largest graph) and uses the arithmetic centroid.
SHAP values are expected-gradient estimates (`shap.GradientExplainer`) over the model
head, with the graph branch collapsed to its pooled embedding. FakeWashington is
`FakeWashingtonV2` (V1 was removed from Qiskit). ibm_osaka and ibm_kyoto are retired;
their calibration snapshots are used for featurization, and new hardware data needs a
current device name.

## Verification

Checked on a 2-core CPU / 8 GB machine against the authors' published measurements,
plus a fresh end-to-end run of the data-collection commands. Outputs are in
`verification/`. Full-scale training (500 epochs on 17 M nodes) was not possible on
that CPU, so model results below use reduced budgets where noted.

| Check | This implementation | Paper |
|---|---|---|
| Unit tests (Fig. 1/4 graph, SupermarQ values, Eq. 1, sample size, GSx, metrics, model, training) | 13 passed | — |
| Global-feature dimension on the 1,510 circuits | 41 | 41 |
| `qwalk-noancilla_9` compiled for FakeSherbrooke: ecr / rz / sx / x | 17,781 / 47,287 / 34,440 / 4,293 | 18,273 / 48,126 / 36,291 / 4,262 (Table 3) |
| RQ1 statistics: Tables 1, 2, 5, 6, 7, 15 | match to the reported precision | |
| RQ1 Table 8 | matches, except two `num_gates` values the paper appears to have transposed | |
| Algorithm-family folds on the device data | sizes 46, 47, 35, 36, 28, 30, 40, 38, 40 | same sizes (§7) |
| Devices, global features only, 10-fold CV, 500 epochs | R² 0.891, MSE 0.363, NMSE 0.109 (MAPE 5.6 %, Spearman 0.92) | R² 0.894, MSE 0.346, NMSE 0.106 (Table 12) |
| Simulators, graphs ≤ 1,000 nodes (1,296 of 3,020 samples), full model, 150 epochs | test R² 0.904, NMSE 0.096 | full data, 500 epochs: R² 0.943 |
| Same subset, global features only | test R² 0.07 | 0.573 (full data) |
| Devices, full model fine-tuned from the subset simulator model, 3 folds × 6 epochs, batch 8 | mean R² 0.706, NMSE 0.294, MAPE 10 % (still improving when stopped) | R² 0.905 (10 folds × 500 epochs) |
| Devices, leave-one-family-out CV, global features only | mean R² −1.71; pooled Spearman 0.78 | R² 0.878 (full features, Table 14) |
| `measure-sim` on 36 fresh MQT Bench circuits × 2 simulators | 72 / 72 measured, Eq. 1 n ≤ 3 for all | — |

Things worth knowing when reading these numbers:

* The simulator measurements contain a few very slow runs of tiny circuits (e.g.
  `dj_16` takes 47.6 s on FakeWashington; `graphstate_9` 19.8 s on FakeSherbrooke), so
  R² on a ~130-sample test split moves a lot depending on which of them land in it.
  MAPE and Spearman's ρ are reported alongside R²/MSE/NMSE for that reason.
* Leave-one-family-out is far harder than the random folds: the global-feature MLP
  extrapolates badly to families whose gate mix it never saw (`qwalk-noancilla` is
  predicted at −4.8 s, `random` circuits at ~14.7 s instead of ~8.5 s). Per-fold R² is
  also computed against the small within-family variance.
* The paper's reported numbers come from model selection on the evaluation fold
  (see "Model selection" above); here every number is on data not used for selection.
