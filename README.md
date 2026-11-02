# PU-GWO experiments: resumable pipeline for the revision

This package reruns every experiment the reviewers asked for and builds all tables and figures from
**logged runs only**. Nothing is interpolated or reconstructed. Every number in the output can be
traced to `runs/<job>/evals.jsonl` (the search log) and `runs/<job>/preds.npz` (the saved predictions).

## 1. Setup (RTX 4090, Linux or WSL2)

```bash
python -m venv venv && source venv/bin/activate
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the Figshare dataset (https://doi.org/10.6084/m9.figshare.1512427.v8). Put the zip files
in a folder such as `~/figshare`; they are extracted automatically.

Free disk space: about **8 GB per worker**. Candidate checkpoints of one batch take up to about
3 GB and are deleted after each batch.

## 2. Commands

```bash
# 1) Build the image cache and the patient-grouped 5-fold split, and freeze the settings.
#    Prints patients and images per class for every partition (this is Table 1).
python run.py prepare --raw ~/figshare --run-dir runs            # default fidelity
# python run.py prepare --raw ~/figshare --run-dir runs --profile paper   # 12/35/50 epochs, ~1.4x slower

# 2) Measure real speed on your GPU and get a time estimate per tier (about 5 minutes).
python run.py benchmark --run-dir runs

# 3) Start workers. Two workers usually use a 4090 better than one; check `nvidia-smi`.
#    Run each in its own tmux/screen session so closing the terminal does not stop it.
python run.py worker --run-dir runs
python run.py worker --run-dir runs        # second worker, second terminal

# 4) Progress, and measured time per run with the remaining estimate.
python run.py status --run-dir runs

# 5) Build all tables and figures from whatever has finished (safe to run at any time).
python run.py analyze --run-dir runs --out results
```

Useful worker options: `--tiers 1 2`, `--methods pu_gwo pareto_gwo`, `--folds 0 1`,
`--retry-failed`, and `--min-free-gb`.

## 3. Interruptions and resuming

You can stop a worker at any time: Ctrl-C, a power cut, a closed terminal, or a crash. Start it
again with the same command.

- Every candidate evaluation is appended to `evals.jsonl` as soon as it finishes. On restart, the
  optimiser is rebuilt from its seed and replays the log, so the search continues exactly where it
  stopped. At most the one candidate that was in training is lost.
- Final retraining saves a checkpoint after every epoch and resumes from it.
- A lock left by a dead worker is detected and reclaimed automatically.
- A full disk stops the worker cleanly without marking the job as failed.
- A job that fails twice for another reason is skipped. Its reason is recorded in
  `runs/<job>/failures.txt`. Use `--retry-failed` after fixing the problem.
- Do not edit `runs/settings.json` or `runs/splits.json` after jobs have started. The run directory
  is one frozen experiment.

## 4. What runs, in priority order

Jobs run tier by tier, and fold by fold inside each tier, so the comparisons that matter most are
completed on all folds first. If time runs out, the finished jobs are still a consistent,
reportable subset.

| Tier | Methods | Answers |
|---|---|---|
| 1 | fixed ResNet-50, PU-GWO, Pareto-GWO, PU-GWO with random Û, PU-GWO with ρ = 0 | The core claim and its key controls (Comments 3, 4) |
| 2 | GWO, NSGA-II, MOTPE, TPE | Same-objective and scalar baselines (Comment 5) |
| 3 | cumulative ablation steps, leave-one-out (no leader weights, no momentum), shuffled Û, NLL-based Û, Random Search, PSO | Ablations and remaining baselines |
| 4 | ρ = 0.25 and 0.75, Pareto-rank Û, extended bound | Sensitivity analyses |

Each job is one method on one fold (5 folds). Seeds are shared across methods for each fold, so
every comparison is paired.

**Time:** run `benchmark` to get a real estimate. My rough expectation for a 4090 is 0.7–1 GPU-hour
per search run, so tiers 1 and 2 take about 22–30 hours with two workers. Tier 3 finishes partly
within 48 hours. If you are short of time, `--tiers 1 2` is the minimum that addresses the
reviewers' main comments.

## 5. Outputs (`results/`)

| File | Paper item |
|---|---|
| `table1_split_composition` | Table 1: patients and images per class, per fold and partition |
| `table2_search_space` | Table 2: search space and all fixed settings |
| `table3_main_results` | Main table: Macro-F1, MCC, NLL, ECE, AURC, E-AURC, error-detection AUROC |
| `table3b_temperature_scaling` | Post-hoc temperature-scaling comparison (Comment 6) |
| `table4_secondary_metrics` | Accuracy, balanced accuracy, precision, recall, and patient-level metrics |
| `table5_risk_at_coverage` | Risk at 100/95/90/80% coverage |
| `table6_cumulative_ablation` | Cumulative ablation |
| `table7_controls` | Random, shuffled, NLL and rank Û controls, plus leave-one-out ablations (Comment 4) |
| `table7b_rho_sensitivity` | ρ ∈ {0, 0.25, 0.5, 0.75}, with ρ = 0 an actual run |
| `table8_class_sensitivity` | Class sensitivity as counts from the confusion matrix pooled over folds |
| `table9_compute_cost` | Measured GPU-hours, epochs trained, and peak memory (Comment 7) |
| `table10_selected_configs` | Selected hyperparameters per method and fold (Comment 8) |
| `table11_paired_statistics` | PU-GWO minus each comparator: patient-bootstrap 95% CIs and Wilcoxon tests (Comment 3) |
| `table_hv_summary` | Final validation hypervolume and evaluations to 95% of each method's own final value |
| `fig_reliability` | Real reliability diagram with a confidence histogram (replaces the old Fig. 3) |
| `fig_hypervolume` | Real hypervolume after every batch (replaces the old Fig. 5b) |
| `fig_risk_coverage` | Full risk–coverage curves |
| `fig_ablation_rho` | Ablation and ρ sensitivity |
| `fig_clipping_analytical` | Analytical clipping figure (Fig. 2; no data involved) |
| `fig_confusion_pugwo`, `fig_temperature_scaling` | Supporting figures |

Every table is written as both `.csv` and booktabs `.tex`. Figures are written as `.pdf` and `.png`.
`ANALYSIS_NOTES.md` lists which method/fold jobs were included. Report only what is there, and state
the number of folds for any method with fewer than 5.

## 6. Protocol details to state in the revised Methods section

These differ from the original manuscript. Describe them as they are:

- **Split.** 5-fold `StratifiedGroupKFold` by patient ID for the test folds (about 20% of patients
  each). The remaining patients are split into train and validation with the same grouping (about
  64/16/20 overall). Each image is tested exactly once across the folds.
- **Fidelity (default profile).** Stage 1 trains for up to 10 epochs (patience 3). Exactly 3 of every
  10 candidates are promoted for every method, and continue to up to 25 epochs (patience 5). The
  cosine learning-rate schedule spans 25 epochs, so stage 1 is a truncation of the stage-2 run.
  Final retraining runs for up to 40 epochs (patience 8). All early stopping and checkpoint
  selection use validation NLL. The `paper` profile uses 12/35/50 epochs instead.
- **Objectives.** Promoted candidates use their stage-2 objective values. The fidelity of every value
  is recorded in the log.
- **Final selection.** Scalar methods pick the best validation Macro-F1. All multiobjective methods
  (NSGA-II, MOTPE, Pareto-GWO, PU-GWO and its variants) pick the point nearest the normalised ideal
  point on the nondominated set of all evaluated candidates.
- **Schedule.** Every GWO variant uses the same cosine schedule for a(t). Standard GWO uses the
  three best candidates found so far as leaders.
- **MC dropout.** Dropout is applied in the classifier head only: 10 passes during search and 30 for
  the final evaluation of every method, including the fixed baseline.
- **Hypervolume.** Computed over all evaluated candidates, after fixed normalisation with bounds
  (1 − F1, NLL, AURC) ∈ [0, 0.25] × [0, 0.8] × [0, 0.08] and reference point 1.1, then scaled to
  [0, 1]. These bounds were fixed before any results were seen; do not tune them afterwards.
- **Pretrained weights.** torchvision's default ResNet-50 ImageNet weights (IMAGENET1K_V2).
- **Fixed baseline.** Learning rate 1e-4, dropout 0.5, weight decay 1e-4, batch size 32, all 4
  stages unfrozen, no label smoothing.
- **Hardware.** The GPU, CUDA version and PyTorch version must be updated to what you actually used.

## 7. Before you run the real experiment

To check the pipeline in a few minutes on synthetic data (CPU is fine):

```bash
python run.py prepare --profile smoke          # writes to runs_smoke/, never mixes with runs/
python run.py worker --run-dir runs_smoke --methods fixed pu_gwo pareto_gwo --folds 0
python run.py analyze --run-dir runs_smoke --out results_smoke
```

Smoke numbers come from random-weight ResNet-18 on synthetic 32-pixel images. They are meaningless
and must never be used in the paper. Delete `runs_smoke/` and `results_smoke/` afterwards.

Whatever the real results turn out to be, report them as they are, including if PU-GWO does not beat
the random-Û control or temperature-scaled baselines. The analysis is designed so that either outcome
is a defensible paper.
