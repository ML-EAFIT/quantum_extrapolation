# Verification outputs

Produced with `qetime` on the authors' published measurements (`qetime fetch-paper-data`)
and on freshly generated circuits; see the "Verification" section of the top-level README.

* `rq1_paper_measurements/` — `qetime analyze` on the four published time files
  (Tables 1, 2, 5–8, 15; histograms as in Figs. 7–8).
* `sim_subset_full_model/` — `qetime train --max-nodes 1000 --epochs 150` on the simulator data
  (test-split metrics, predictions, SHAP importance of the global features).
* `sim_subset_global_only/` — same with `--global-only`.
* `hw_10fold_global_only/` — `qetime cv --global-only` on ibm_osaka/ibm_kyoto data, 10 folds, 500 epochs.
* `hw_family_cv_global_only/` — same with `--by-family` (leave-one-algorithm-family-out).
* `hw_full_model_reduced/` — `qetime cv --pretrained <subset simulator model> --folds 3 --epochs 6 --batch-size 8`.
* `smoke_pipeline/` — `generate-circuits` → `measure-sim` (36 circuits × FakeSherbrooke/FakeWashington,
  3 repeats each) → `select` (GSx) → `train` → `cv --by-family`, on 3–8-qubit circuits.
