# Results and honest findings

## Start here

- [`VERSION_HISTORY.md`](VERSION_HISTORY.md): the four-stage research sequence.
- [`Goal_Refinement_v5_Full_History.png`](Goal_Refinement_v5_Full_History.png): metrics over all 1,480 batches. The run consists of consecutive stages, so it should not be interpreted as a single continuous learning curve.
- [`checkpoint_extracted/Goal_Refinement_v5_Full_Checkpoint_History.csv`](checkpoint_extracted/Goal_Refinement_v5_Full_Checkpoint_History.csv): the complete machine-readable history.

## What the v5 checkpoint actually shows

| Finding | Evidence |
|---|---|
| Training cost decreased consistently | Mean cost fell from 1.449 over the first 100 batches to 1.209 over the last 100. |
| Position tracking improved modestly | Validation position RMSE moved from 0.154 m to 0.143 m. |
| Velocity tracking barely improved | Validation velocity RMSE moved from 0.622 m/s to 0.613 m/s. |
| Overall generalization did not improve | Validation total cost was best at batch 20 and ended worse at batch 1,480. This may be related to differences in validation duration and deterministic evaluation, but the current evidence does not isolate the cause. Because validation cost rose early, conventional late-stage overfitting alone does not clearly explain the pattern. |
| The critic became predictive | Final explained variance was 0.972 in training and 0.926 in validation, indicating that the critic fit the observed returns well. |

## Existing detailed figures

The original goal-refinement figures are under `training_history/goal_refinement/`:

- tracking metrics;
- residual feasibility;
- PPO/critic learning health;
- exploration standard-deviation sweep.

Early supervised-controller histories are under `training_history/early_training/`.
