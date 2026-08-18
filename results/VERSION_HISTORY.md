# Version history

The project is shown as a research progression rather than presenting only the newest unfinished model.

## v1: supervised torque prediction

The first system learned actuator commands from state error. It generated random torque and train the model so that given the position prior the random torque and after the random torque, match the torque sequence it had. Even though this supervised learning lead to success, the generlization to the meaningful torque sequences was not successful.

## v2: continuous-goal reinforcement learning

The project moved from direct torque supervision to PPO policies for reaching single and repeated goals. There was big collapse caused by learning.

## v3: motion-sequence reinforcement learning

Short goal windows became recurrent, variable-duration motion rollouts with action and recurrent-state carriers. There were some degree of success, however long term motions were unstable. I think this caused because of the learning proccess itself, which forced the duration it have to reach the goal in, or just a lack of ability for machine learning to learn to the perfection

## v4: residual goal refinement

The focus shifted to the data itself: CMU retargeting, shoulder discontinuities, spine allocation, joint clipping, collision correction, Hermite targets, PD gains, and torque feasibility.

A recurrent PPO actor proposed joint-position and joint-velocity residuals while a critic estimated causal rollout cost. The preserved checkpoint reached batch 1,480. Tracking improved modestly, but total held-out cost did not improve.

I am still searching for the reason why the cost not decreased significantly, or the problem of sample efficiency.
