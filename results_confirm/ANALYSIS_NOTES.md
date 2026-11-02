# Analysis notes

Completed jobs: 45

- ResNet-50 (fixed): folds [0, 1, 2, 3, 4]
- GWO: folds [0, 1, 2, 3, 4]
- MOTPE: folds [0, 1, 2, 3, 4]
- NSGA-II: folds [0, 1, 2, 3, 4]
- Pareto-GWO: folds [0, 1, 2, 3, 4]
- PU-GWO: folds [0, 1, 2, 3, 4]
- R3: rank Û + AURC-weighted leaders: folds [0, 1, 2, 3, 4]
- PU-GWO, random Û: folds [0, 1, 2, 3, 4]
- PU-GWO, ρ=0 (no feedback): folds [0, 1, 2, 3, 4]

All values are computed from preds.npz and evals.jsonl. ECE in the reliability figure is computed on pooled predictions, so it differs slightly from the fold mean in Table 3. With 5 folds the smallest possible two-sided Wilcoxon p is 0.0625; rely on the bootstrap CIs for uncertainty.
