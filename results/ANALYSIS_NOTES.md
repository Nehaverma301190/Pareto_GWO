# Analysis notes

Completed jobs: 100

- Pareto-GWO + AURC movement: folds [0, 1, 2, 3, 4]
- ResNet-50 (fixed): folds [0, 1, 2, 3, 4]
- GWO: folds [0, 1, 2, 3, 4]
- MOTPE: folds [0, 1, 2, 3, 4]
- NSGA-II: folds [0, 1, 2, 3, 4]
- Pareto-GWO: folds [0, 1, 2, 3, 4]
- PSO: folds [0, 1, 2, 3, 4]
- PU-GWO, extended bound: folds [0, 1, 2, 3, 4]
- PU-GWO: folds [0, 1, 2, 3, 4]
- PU-GWO, NLL-based Û: folds [0, 1, 2, 3, 4]
- PU-GWO − leader weights: folds [0, 1, 2, 3, 4]
- PU-GWO − momentum: folds [0, 1, 2, 3, 4]
- PU-GWO, random Û: folds [0, 1, 2, 3, 4]
- PU-GWO, Pareto-rank Û: folds [0, 1, 2, 3, 4]
- PU-GWO, ρ=0 (no feedback): folds [0, 1, 2, 3, 4]
- PU-GWO, ρ=0.25: folds [0, 1, 2, 3, 4]
- PU-GWO, ρ=0.75: folds [0, 1, 2, 3, 4]
- PU-GWO, shuffled Û: folds [0, 1, 2, 3, 4]
- Random Search: folds [0, 1, 2, 3, 4]
- TPE: folds [0, 1, 2, 3, 4]

All values are computed from preds.npz and evals.jsonl. ECE in the reliability figure is computed on pooled predictions, so it differs slightly from the fold mean in Table 3. With 5 folds the smallest possible two-sided Wilcoxon p is 0.0625; rely on the bootstrap CIs for uncertainty.
