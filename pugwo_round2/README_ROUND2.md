# Round 2: redesigned PU-GWO (development + confirmation)

Copy these five files into `D:\pugwo_experiments\pugwo_experiments\`, replacing the old ones:
`config.py`, `optimizers.py`, `run.py`, `analyze.py`, `dev_compare.py`.
Back up the old folder first. The original methods behave exactly as before.

## The three redesign candidates

| Name | Change | Why |
|---|---|---|
| `pu_r1_rank` | Û from the **rank** of AURC in the population, not min-max of its value | Validation AURC rests on a handful of errors; min-max turns tiny noisy gaps into a 0-1 spread. Ranks are robust to that. |
| `pu_r2_lead` | **Leaders** (alpha/beta/delta) are drawn favouring archive members with low AURC; no step-size feedback | Steers the pack towards configurations that rank errors well, instead of only adding exploration to poor candidates. |
| `pu_r3_both` | R1 + R2 | Combination. |

The leader weighting strength (kappa = 2.0) and rho = 0.5 are fixed in advance and are not tuned.

## Step 1: development (about 1.5-2 hours with three workers)

Runs the three candidates on the original 5 folds (15 jobs). Start three workers:

```bat
python run.py worker --run-dir runs --tiers 5
```

**Do NOT run `analyze` on `runs` until the confirmation is complete.** It would show test results for
the new variants, and the design choice must not be influenced by test data.

## Step 2: choose the design (validation only)

```bat
python dev_compare.py --run-dir runs
```

It reads validation predictions only and applies the decision rule written at the top of
`dev_compare.py` (fixed before any run):

- pick the candidate with the lowest mean validation AURC,
- it must beat BOTH the original PU-GWO and rho = 0 on validation AURC,
- and its validation Macro-F1 must be within 0.5 points of rho = 0.

If it prints `SELECTED for confirmation: <name>`, write that name and today's date in a file
`DESIGN_CHOICE.txt` before Step 3. If no candidate qualifies, stop: the paper is written from the
first experiment (option A).

## Step 3: confirmation on a fresh patient split (about 4-5 hours with three workers)

```bat
python run.py prepare --profile confirm
```

This creates `runs_confirm` with a new patient split (split seed 7) and new training seeds (101),
reusing the existing image cache. Then start three workers, replacing `<CHOSEN>`:

```bat
python run.py worker --run-dir runs_confirm --methods fixed pareto_gwo pu_gwo pu_rho0 pu_random_u nsga2 motpe gwo <CHOSEN>
```

That is 45 jobs. Check progress with `python run.py status --run-dir runs_confirm`.

## Step 4: results

```bat
python run.py analyze --run-dir runs_confirm --out results_confirm --ref <CHOSEN>
```

The paired statistics then compare the chosen design against every other method. Afterwards you can
also run `python run.py analyze --run-dir runs --out results` again to include the development
variants in the first-experiment tables.

Whatever the confirmation shows, report it, together with the first experiment and the
decision rule above.
