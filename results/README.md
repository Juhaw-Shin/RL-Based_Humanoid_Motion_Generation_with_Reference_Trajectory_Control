# Results and honest findings

## Start here

- [`VERSION_HISTORY.md']: the four-stage research sequence.
- [`Goal_Refinement_v5_Full_History.png`](Goal_Refinement_v5_Full_History.png): all 1,480 batches's metrics graph throughout the run. THe run is made of consequtive stages, so should not be interpreted as a continous learning curve.
- [`checkpoint_extracted/Goal_Refinement_v5_Full_Checkpoint_History.csv`](checkpoint_extracted/Goal_Refinement_v5_Full_Checkpoint_History.csv): the complete machine-readable history.

## What the v5 checkpoint actually shows

| Finding | Evidence |
|---|---|
| Training optimized something consistently | Mean cost fell from 1.449 over the first 100 batches to 1.209 over the last 100. |
| Position tracking improved modestly | Validation position RMSE moved from 0.154 m to 0.143 m. |
| Velocity tracking barely improved | Validation velocity RMSE moved from 0.622 m/s to 0.613 m/s. |
| Overall generalization did not improve | Validation total cost was best at batch 20 and ended worse at batch 1,480. This maybe is due to the difference in duration of validation and also the fact that the validation run was deterministic, however this shows that somehow the duration or sampling related problem occurred. I Don't think this is due to overfitting as the rise in validation cost were almost from the start.|
| The critic became predictive | Final explained variance was 0.972 in training and 0.926 in validation. This also indicates that the learning became too slow and therefore predictable.|

## Existing detailed figures

The original goal-refinement figures are under `training_history/goal_refinement/`:

- tracking metrics;
- residual feasibility;
- PPO/critic learning health;
- exploration standard-deviation sweep.

Early supervised-controller histories are under `training_history/early_training/`.
