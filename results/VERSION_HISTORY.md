# Version History

This project evolved through several modeling approaches. Earlier versions are documented because failed approaches directly motivated the current architecture.

## v1: Supervised Torque Prediction

The first approach attempted to learn actuator torques directly from state transitions. Random torque sequences were applied to the humanoid, and a supervised model was trained to recover the corresponding torque sequence from the states before and after the motion.

The model learned the supervised training task, but it did not generalize reliably to meaningful long-horizon motion. This motivated moving away from direct torque prediction.

## v2: Continuous-Goal Reinforcement Learning

The project shifted from supervised torque prediction to PPO-based policies for reaching individual and repeated target states.

Training suffered from policy collapse and unstable learning, which motivated changes to both the action representation and training procedure.

## v3: Motion-Sequence Reinforcement Learning

The target representation was extended from individual goals to recurrent, variable-duration motion sequences. The policy received short future goal windows together with recurrent state.

This produced partial success on shorter motions, but longer-horizon behavior remained unstable. Possible causes included the difficulty of simultaneously learning motion timing, trajectory structure, and low-level actuation.

## v4/v5: Reference Motion and Residual Goal Refinement

The project shifted toward separating trajectory generation from physical control.

Motion data were retargeted and corrected for issues including shoulder discontinuities, spine allocation, joint clipping, self-collision, interpolation, PD gains, and torque feasibility.

A supervised trajectory planner generates reference motion from initial and goal states. A recurrent PPO actor then proposes joint-position and joint-velocity residuals to refine the reference, while a PD controller converts the resulting targets into actuator torques.

The preserved residual-refinement run reached 1,480 training batches. Position tracking improved modestly, but total held-out validation cost did not improve.

The current research focus is understanding the source of this behavior and improving the sample efficiency and stability of residual refinement.
