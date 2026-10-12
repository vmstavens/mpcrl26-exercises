# Case Study: Cross-Simulator Transfer and MPC for Cable Alignment

## Project Summary

**Working title:** Cross-Simulator Transfer and MPC Comparison for Deformable Cable Alignment.

**Primary research questions:** Does a frozen MuJoCo-trained alignment controller transfer to a PyElastica cable environment without retraining? How does standalone MPC compare with that transferred controller on the same Elastica alignment task?

**Control objective:** A better smoothness-performance tradeoff, quantified by RMS and peak translational and angular gripper acceleration, cable oscillation, alignment success, and completion time. Moving slowly or doing nothing does not count as improvement. Jerk is an optional secondary metric when measurements are reliable.

The primary study has two parts: replace the MuJoCo cable dynamics with a PyElastica Cosserat-rod environment and evaluate the frozen controller; then implement standalone MPC and compare the controllers within Elastica. Keep all existing controller networks and normalization statistics frozen. Building or calibrating the MPC model does not retrain the RL agent.

This is different from the earlier action-correction proposal: standalone MPC chooses actions independently of DRLR. The earlier correction layer remains an optional hybrid extension, with its mathematical and implementation details retained below.

```text
Experiment A: Elastica -> observation adapter -> frozen alignment controller -> Elastica
Experiment B: Elastica -> state/model interface -> standalone MPC -> Elastica
Optional C:  Elastica -> frozen controller -> MPC correction -> Elastica
```

MPC predicts a short sequence, executes only its first action, and replans after receiving the next observation. Transfer performance and MPC benefits are hypotheses to test, not assumptions.

## Scope and Non-Goals

Start with contact-free cable-tip alignment to a target frame corresponding to the pipe entrance. The first Elastica environment need not reproduce the complete pipe collision geometry. Use nominal material properties, a fixed target pose, and noiseless observations. Retain the existing alignment thresholds, time limit, and success-persistence rules.

The minimum project does not include:

- Retraining DP, SAC, RMA, or the critics.
- Changing DRLR's critic-based selection or Bellman targets.
- Learning a complete cable simulator or adding perception.
- Planning full contact-rich insertion.
- Claiming collision avoidance, stability, or formal safety guarantees.
- Claiming real-time operation before measuring end-to-end latency.

Later extensions can investigate material variation, observation noise, RMA-conditioned prediction, and MPC-assisted RL training separately.

## Primary Study: Elastica Transfer and Standalone MPC

### Part 1: Evaluate the Frozen Alignment Controller

Implement an Elastica environment behind an adapter that preserves the checkpoint's observation and action contracts. Begin with the controller currently used for evaluation and record exactly what it includes:

- The trained SAC actor alone, or the full DRLR teacher/actor/critic selector.
- Privileged phase-1 RMA, or history-based phase-2 adaptation.
- The original observation normalizer, EMA state, action history, and motion limits.

These are distinct controllers. Do not silently replace the full selector with actor-only deployment and describe them as the same evaluation. For the full selector, retain the DP observation history and critics' expected inputs as well.

The environment adapter must reproduce observation ordering, entrance-relative coordinates, tip direction convention, TCP pose, cable-keypoint sampling, and normalized action scaling. If the checkpoint uses the legacy 60D observation, construct that contract instead of assuming the native 48D layout. Supply geometry needed by observations even if downstream pipe contact is omitted.

Drive the grasped end with equivalent translation and rotation increments and reproduce the grasp-frame offset. In Elastica this requires an appropriate moving boundary/attachment; it is not a MuJoCo mocap call. Distinguish an ideal prescribed grasp pose from an attachment with compliance or tracking lag, and document any mismatch.

Use a stable internal integration timestep with enough substeps per policy control interval. Preserve physical time and policy update frequency; changing the physics timestep does not authorize changing the action semantics or controller clock.

**Deliverable:** Frozen-policy rollouts in Elastica, with success, motion metrics, and traces comparable to the original MuJoCo baseline.

### Gate: Validate the Environment Before Judging Transfer

Replay prescribed small translations, rotations, and hold-still sequences before running learned control. Check direction signs, units, frame offsets, gravitational sag, transient response, and boundary behavior. Compare nominal cable responses between MuJoCo and Elastica using the same physical inputs.

Calibrate geometry, total mass, stiffness, damping, rest shape, and grasp conditions. Equal parameter names or values do not establish equivalent dynamics; for example, volumetric density and linear density must be converted using cross-sectional area, and damping definitions may differ.

Check mesh and timestep convergence. Resample centerlines consistently rather than assuming MuJoCo body centers are Elastica rod nodes. Shape observations do not uniquely determine cross-section twist, internal strain, or velocity, so reconstructing arbitrary dynamic initial states may require estimation. Start from reproducible, physically consistent settled configurations where possible.

Only after these checks interpret transfer failures as evidence of controller sensitivity to a dynamics change. Elastica is an alternative simulator, not experimentally established physical ground truth for the actual cable.

### Part 2: Standalone MPC in Elastica

Build an MPC controller that independently chooses gripper increments toward the same alignment target. Its conceptual dynamics are

$$
\hat x_{k+1}=f_{\mathrm{rod}}(\hat x_k,u_k;\eta),
$$

where `x` includes rod shape and motion, `u` is the grasp movement, and `eta` contains material and geometric parameters. The controller may use direct Elastica rollouts or a calibrated reduced dynamic model. Benchmark direct previews before assuming they are affordable; a PyElastica simulator is not automatically a CasADi/acados model with derivatives.

An initial standalone objective is

$$
J_{\mathrm{MPC}}=\sum_{k=1}^{H}\|\hat y_k\|_Q^2+
\sum_{k=0}^{H-1}\left(\|a_k\|_U^2+\|a_k-a_{k-1}\|_T^2\right).
$$

`y` is tip position/direction error, `a` is normalized gripper movement, `U` penalizes movement magnitude, and `T` penalizes abrupt changes. Use the same physical limits as the transferred controller. Unlike the optional correction formulation below, standalone MPC has no penalty for deviating from a DRLR nominal action and no DRLR-centered correction radius.

Execute the first action and replan. Use a tested fallback on solver failure. For a direct rollout approach, use an established sampling-based planning implementation; for a reduced symbolic model, use the CasADi/acados workflow. Avoid initially attempting full contact-rich nonlinear rod optimization at the original control deadline.

**Deliverable:** Standalone MPC rollouts on the same Elastica task, with prediction quality and end-to-end computation cost reported.

### Model and State Access

An MPC predictor that uses the exact Elastica plant and full internal state has information unavailable to an observation-based policy. Label this condition **model-informed MPC with simulator-state access**, or an oracle diagnostic. Comparing it with zero-shot transferred RL does not demonstrate that MPC is generally superior to RL.

For a deployment-oriented comparison, initialize MPC from observable state estimates and use a calibrated reduced or deliberately mismatched predictor. Record each controller's access to hidden material parameters, rod velocities, orientations, and strains. Match or explicitly distinguish information access; do not present an oracle-state result as an observation-matched comparison.

Keep calibration/tuning data separate from final evaluation episodes. Report their volume and any use of Elastica data, since the frozen policy has had no Elastica retraining. Matching runtime limits does not remove this adaptation-data asymmetry.

### Suggested Sequence

1. Reproduce the selected frozen-controller baseline in MuJoCo.
2. Build and validate the Elastica environment adapter with prescribed motions.
3. Run zero-shot controller transfer under calibrated nominal conditions.
4. Implement standalone MPC and compare controllers within Elastica.
5. Test controlled stiffness or damping changes without policy retraining.
6. Optionally combine the frozen controller with MPC correction.

Part 1 is an independent milestone. If adapter/model construction uses the available school time, a well-validated transfer study is a useful partial result, but the full two-part study requires the MPC comparison.

## Existing Implementation

The case study uses the neighboring `dlo-manipulation-lab` repository. The paths below refer to that repository; they describe existing integration points, not implemented MPC functionality.

| Existing component | Role in the case study |
| --- | --- |
| `learning/reinforcement/compiled_drlr.py`, `_environment_step()` | Produces teacher and actor candidates, selects an action, and advances the environment. |
| `learning/reinforcement/compiled_drlr.py`, `_exact_candidate_selection_diagnostics()` | Implements same-state twin-critic selection in the relevant mode. |
| `learning/reinforcement/brax_runner.py`, `NormalizedActionEnvironment` | Converts normalized actions to physical increments. |
| `learning/action_motion_limits.py`, `limit_action()` | Existing speed/acceleration limiter and matched baseline. |
| `learning/reinforcement/rma.py`, `full_state60_to_target_align48()` | Defines entrance-frame tip, TCP, and cable features. |
| `envs/grasped_end_dlo/target_align/task.py` | Defines position/direction alignment and success conditions. |
| `envs/grasped_end_dlo/base.py`, `step()` and `_apply_action()` | Executes physical mocap translation and rotation increments. |

There are multiple task and controller configurations. Load the exact checkpoint's resolved configuration instead of assuming every run uses the same observation layout, action filtering, or target predicate.

The first prototype should use a standalone evaluation loop with one environment. A Python CasADi solver cannot simply be inserted into the existing JAX-compiled scan. Compiled integration is later engineering work.

## Optional Correction: Mathematical Formulation

The following retains the original hybrid proposal for optional experiment C. It is not the mandatory formulation for standalone MPC in experiment B. Its alignment features, action units, and physical motion constraints can also inform the primary study.

### Observed Alignment Error

For the 48D entrance-frame observation, the first three entries are tip position and the next three are tip direction. Define

$$
y_t = \begin{bmatrix}p_t-p^\star\\d_t-d^\star\end{bmatrix}\in\mathbb{R}^6.
$$

Here `p_t` is the tip position in meters and `d_t` is its unit direction. For ordinary entrance alignment, `p_star = [0, 0, 0]` and `d_star = [0, 0, 1]`. If the task uses a target offset or pre-entry region, choose a reference consistent with that configuration. Do not change the environment's success predicate to favor the controller.

The task aligns a direction, not full orientation: cable-tip roll is not part of this error. For unit directions separated by angle `theta`,

$$
\|d_t-d^\star\|_2^2=2(1-\cos\theta)\approx\theta^2
$$

near alignment. This gives a convenient local orientation cost without predicting a full rotation matrix.

### Actions and Coordinates

Use the normalized action `a_t` supplied to `NormalizedActionEnvironment`. With symmetric physical bounds,

$$
a_t\in[-1,1]^6,\qquad u_t=S a_t,
$$

where `S` is a diagonal matrix of translation and rotation increment scales. For general bounds, retain the actual affine mapping `u = low + 0.5 * (a + 1) * (high - low)`.

The first three physical components are translation increments; the last three are rotation-vector increments. They are not velocities or absolute gripper poses.

For example, `configs/action/grasped_end_dlo_mocap_delta_stride3_realtime_v2.yaml` specifies per-component bounds of 0.001 m and 0.01 rad and a 0.002 s control interval. These are example values, not universal defaults. Dataset subsampling stride must not be substituted for the environment's control interval.

Initially fix the pipe orientation so a model fitted to world-frame actions and entrance-frame features remains consistent. For randomized pipe orientations, either transform physical translation/rotation increments into the entrance frame consistently or condition the model on the frame orientation.

### Reduced Prediction Model

Begin with a linear autoregressive model:

$$
\hat y_{t+1}=A y_t+D y_{t-1}+B a_t+c,
\qquad A,D,B\in\mathbb{R}^{6\times6},\quad c\in\mathbb{R}^6.
$$

The previous error supplies information about motion and lag. With `x_t = [y_t; y_(t-1)]`, the model is a 12D discrete state-space system:

$$
x_{t+1}=\begin{bmatrix}A&D\\I&0\end{bmatrix}x_t+
\begin{bmatrix}B\\0\end{bmatrix}a_t+
\begin{bmatrix}c\\0\end{bmatrix}.
$$

This is an approximation, not an assertion that tip history fully describes the cable. Different shapes, velocities, and contacts can produce different responses at the same tip pose. Add TCP features, selected cable keypoints, or longer history if validation shows that the reduced model is insufficient.

Linear predictions can also drift away from unit-length directions. Keep the horizon and corrections local and monitor this error. Normalizing predicted directions inside the optimization would change the formulation into a nonlinear problem.

### Identification Objective

Collect samples `(y_t, y_(t-1), a_t, y_(t+1))` and fit

$$
\min_{A,D,B,c}\sum_t\|y_{t+1}-Ay_t-Dy_{t-1}-Ba_t-c\|_W^2+
\lambda(\|A\|_F^2+\|D\|_F^2+\|B\|_F^2).
$$

`W` scales position and direction residuals; alternatively standardize features and perform ridge regression in the scaled coordinates. Save every scaling statistic with the model. Fit through a structured least-squares/ridge implementation, not an explicit matrix inverse.

Use the action actually passed to the normalized environment after selection and filtering. Small bounded perturbations around nominal actions help identify responses that the policy alone might rarely excite. Keep collection within the chosen operating region.

### Receding-Horizon Optimization

At real step `t`, let `a_bar_t` be the action selected by the frozen controller. Initially use `a_bar_k = a_bar_t` as the nominal reference across the planning horizon. This is a reference approximation, not a commitment to hold the policy action in the actual rollout.

Optimize `a_0, ..., a_(H-1)` and their predicted states with

$$
\min J=\sum_{k=1}^{H}\|\hat y_k\|_Q^2+
\sum_{k=0}^{H-1}\left(\|a_k-\bar a_k\|_R^2+
\|a_k-a_{k-1}\|_T^2\right),
$$

subject to the identified dynamics and the measured initial history. Here `a_(-1)` is the previous executed action, and `||v||_M^2 = v^T M v`.

- `Q` rewards predicted tip alignment.
- `R` keeps corrections close to the learned controller.
- `T` discourages abrupt changes in motion.

Choose positive weights and normalize units. One useful parameterization is `Q = diag(q_p / p_scale^2, ..., q_d / d_scale^2)`, with three position and three direction entries. `p_scale` is a meaningful position error; `d_scale` can be derived from a meaningful angular error using `2 * sin(theta_scale / 2)`.

Do not introduce the learned critic into the first MPC objective. A geometric objective makes it easier to distinguish predictive control effects from critic-estimation effects. It may differ from the environment's sparse reward; report task success as well as the geometric metric.

### Constraints

Use normalized action bounds and a componentwise correction radius:

$$
-1\le a_k\le1,\qquad -\rho\le a_k-\bar a_k\le\rho.
$$

Match physical speed and acceleration limits where configured. For the translational group,

$$
\|S_p a_k^p\|_2\le v_{\max}\Delta t,\qquad
\|S_p(a_k^p-a_{k-1}^p)\|_2\le\alpha_{\max}\Delta t^2.
$$

Use analogous angular constraints. These bound incremental motion, not contact forces or continuous-time cable safety. Norm constraints match the vector-norm semantics of the existing limiter; independent axis bounds are not equivalent.

Linear dynamics, quadratic costs, and affine/box constraints form a QP. Exact norm limits add convex conic constraints, even if a prototype expresses them through a general nonlinear solver. Optional workspace constraints require predicting TCP position too; do not enforce them using tip position as a substitute.

## Optional Correction: Implementation Plan

These stages implement the optional DRLR-plus-MPC correction layer. Follow the primary-study sequence above first; standalone MPC does not require a nominal action from DRLR.

### Stage 0: Establish the Baseline

- Select one checkpoint and record its task, observation contract, normalizer, action scales, filter settings, and RMA mode.
- Run a small frozen-controller evaluation and save initial seeds, success, alignment errors, and action traces.
- Reproduce those rollouts before adding correction. Preserve the existing success-persistence and termination rules.
- Begin with the phase-1 simulation controller if convenient, but label its privileged inputs explicitly. Do not describe it as history-only deployment.

**Deliverable:** A repeatable evaluation harness and an unchanged-controller baseline.

### Stage 1: Collect Identification Data

- Record tip features, relevant full observations, nominal actions, executed normalized and physical actions, episode IDs, and timestamps/control intervals.
- Collect unmodified policy rollouts and bounded perturbed rollouts; label their provenance separately.
- Preserve pre-reset terminal observations where available. Never use an automatically reset observation as the next state of a terminating transition.
- Avoid or label contact samples. Do not train a pre-contact model on untracked mixtures of contact and free-motion behavior.
- Split data by episode into training, validation, and final held-out evaluation sets.

**Deliverable:** A versioned dataset with exact coordinate, action, and timing conventions.

### Stage 2: Validate the Predictor

- Fit the linear model and compare against persistence and constant-motion prediction baselines.
- Evaluate open-loop multi-step predictions under recorded actions, matching the intended MPC horizon.
- Report position error, direction-angle error, direction-norm drift, and errors by operating region.
- Validate bounded perturbed actions, not only the unmodified policy's action distribution.
- Select horizon and model features using validation data, never final test episodes.

**Gate:** Continue to closed-loop MPC only when the predictor adds useful information at the planning horizon. A poor model is a result to document, not something a more sophisticated solver fixes.

### Stage 3: Build the MPC Prototype

Use CasADi `Opti` for readability. Declare state and action trajectories as decision variables, and measured initial history, previous executed action, and nominal sequence as parameters. Construct the problem once; update numerical parameter values at each interaction.

A general solver such as IPOPT is acceptable for an initial offline simulation prototype. Do not equate a successful solve with meeting the control deadline. Shift the previous solution for warm starting and reset warm-start/history state at episode boundaries.

Return the first action plus diagnostics: solve status, iterations, solve time, predicted error trajectory, objective terms, correction magnitude, and constraint residuals. Check finite values and feasibility before execution.

If solving fails or exceeds a chosen deadline, use a prechecked feasible nominal action. If it is not feasible, use a separately defined bounded fallback compatible with previous motion; blindly applying zero can violate acceleration limits. Record every fallback and intervention.

**Deliverable:** A controller with tested dynamics, constraints, warm starts, resets, and failure handling.

### Stage 4: Connect the Evaluation Loop

The following illustrates proposed interfaces, not existing APIs:

```python
nominal, controller_state = frozen_controller.propose(observation, controller_state)
action, diagnostics = mpc.correct(error_history, previous_executed, nominal)
next_state = environment.step(state, action)
controller_state = update_executed_history(controller_state, observation, action)
```

Build the nominal action after the existing controller's selection and relevant filtering. MPC must model the final executed action. Do not apply an unmodeled second EMA or clipping stage after the solve; if a final enforcement stage changes the action, log it and feed the actual result into histories and identification data.

Distinguish previous executed action from any internal actor/reference EMA state. Preserve each state according to its actual contract. For history-based RMA deployment, update adaptation history with the action actually applied, in the normalization/domain expected by its export.

Initially run outside `jax.jit`/`jax.lax.scan`. Device-to-host transfers and policy generation are part of end-to-end cost. Avoid refactoring the shared training pipeline merely to support the first evaluation experiment.

**Deliverable:** Frozen-controller rollouts with optional correction and complete diagnostics.

### Stage 5: Timing and acados

Choose the horizon in physical time: `prediction_duration = H * control_dt`. With a 2 ms interval, 10 steps cover only 20 ms. Inspect cable-response measurements before deciding whether that is useful.

For slower MPC updates or move-blocked controls, model the actual repeated-action execution and keep the baseline's update schedule matched. Do not silently change a 2 ms controller into a 20 ms controller and attribute the difference solely to MPC.

After the prototype is validated, use an acados discrete model with `disc_dyn_expr` and `integrator_type = "DISCRETE"`. Augment the 12D predictive state with the previous 6D action if representing action-difference costs and constraints through standard stage-local state/control expressions. The augmented state then has 18 dimensions.

Keep adaptation/model coefficients and nominal references as runtime parameters where appropriate. Benchmark the complete observation-to-action path, not only the solver. A deployment inside the compiled JAX loop needs a supported compiled interface or a redesigned execution boundary; it is not an ordinary Python function call.

**Deliverable:** Either measured real-time feasibility or a clearly labeled slower-than-real-time simulation study.

## Evaluation Design

### Baselines and Ablations

The primary study compares:

| Controller and environment | What it tests |
| --- | --- |
| Frozen alignment controller in MuJoCo | Original performance reference. |
| Same frozen controller in Elastica | Zero-shot cross-simulator transfer. |
| Frozen controller with tuned EMA in Elastica | Whether ordinary smoothing improves the tradeoff. |
| Standalone MPC in Elastica | Model-based control versus the transferred controller on the same task. |
| Frozen controller plus MPC correction in Elastica, optional | Whether combining the controllers adds value. |

Within Elastica, match initial configurations, target definitions, motion limits, physical-time horizons, and controller update schedules. Across simulators, use equivalent geometric/material scenarios and document initialization differences; their internal state vectors need not be identical.

Compare MuJoCo and Elastica results to characterize transfer. Compare RL and MPC results within Elastica to characterize control performance. Do not interpret a change in environment and controller simultaneously as a controller-only effect.

### Correction Ablations (Optional)

For the optional hybrid extension, retain these ablations:

| Condition | What it tests |
| --- | --- |
| Frozen DRLR(DP)+RMA with existing settings | Whether correction improves the current system. |
| Frozen controller with matched physical limits | Whether improvements are explained by motion limiting alone. |
| Ordinary EMA smoothing with matched limits | Whether prediction provides a better smoothness-performance tradeoff than smoothing alone. |
| Full MPC correction | Combined effect of prediction, alignment objective, and bounded corrections. |
| MPC with alignment weight zero | Whether the geometric prediction term provides the claimed benefit. |

For the no-alignment ablation, retain the same model, horizon, limits, and other objective terms. Avoid adding redundant baselines if the existing controller already supplies exactly the same filter/limits.

Tune EMA strength and MPC weights on the same validation episodes under the same physical limits. Compare smoothness-performance curves, not only one weak smoothing setting against a tuned MPC controller. Fix the settings and primary comparison before final held-out evaluation.

Compare paired initial conditions and identical environment randomization/noise protocols across conditions. Policies and histories will diverge after different actions; pairing means equal starting conditions, not identical later observations. Use a separate validation set for weights, horizon, and correction radius.

Start with a small debugging batch, then choose a larger held-out evaluation budget appropriate to available compute. Report episode count, checkpoint count, seeds, variability, and confidence intervals. With one checkpoint, limit conclusions to that controller rather than claiming generality across training runs.

### Metrics

- Per-episode RMS and peak translational acceleration in m/s^2.
- Per-episode RMS and peak angular acceleration in rad/s^2, reported separately from translation.
- Optional translational and angular jerk, describing how abruptly acceleration changes.
- Cable oscillation during a predefined final-alignment/holding window, measured using consistently sampled keypoint velocities or detrended position fluctuations. Report task-progress motion separately; low velocity alone can mean lack of progress.
- Success rate under the existing task predicate and time limit.
- Steps/time to success and time-integrated alignment error.
- Position error and angular error on both successful and failed episodes.
- Physical action magnitude, changes between actions, and limit violations.
- Correction magnitude and frequency of substantial interventions, for the optional hybrid controller.
- Model one-step and multi-step error on states visited by the corrected controller.
- Solver failures, fallback frequency, and constraint residuals.
- Solver latency and end-to-end latency, including tail latency.
- Identification-data volume, so the added data cost is visible.

If task success and smoothness trade off, report the tradeoff rather than choosing only the favorable metric. Gains in predicted cost without gains in actual behavior indicate model exploitation or objective mismatch.

### Measuring Smoothness

Measure motion from the actual simulated gripper/TCP pose or velocity, using a consistent coordinate frame and the actual control interval. If Elastica uses an ideal prescribed grasp boundary without a simulated actuator, label its motion as prescribed boundary motion rather than implying measured robot tracking dynamics. Normalized action differences alone are not a physical smoothness metric. The corresponding commanded increment changes can be reported separately as a control-signal proxy.

For measured translational velocity `v_t`, compute acceleration and its per-episode RMS as

$$
b_t=\frac{v_t-v_{t-1}}{\Delta t},\qquad
b_{\mathrm{RMS}}=\sqrt{\frac{1}{N}\sum_t\|b_t\|_2^2}.
$$

Peak acceleration is `max_t ||b_t||_2`. Apply the same construction to measured angular velocity in a consistent frame. If deriving angular velocity from orientations, use relative rotations rather than subtracting Euler angles. Optional jerk is `(b_t - b_(t-1)) / control_dt`.

Exclude reset discontinuities and report any filtering used for derivative estimates; apply identical measurement processing across controllers. Do not pad completed episodes with stationary samples to lower RMS. Summarize per-episode metrics over matched initial conditions, including failed episodes.

### Proposed Quantitative Targets

These are initial targets to agree on before final evaluation, not expected results or guarantees. For standalone MPC versus the frozen controller within Elastica, with matched motion limits, aim for:

- At least 20% lower mean per-episode RMS translational acceleration and at least 20% lower mean per-episode RMS angular acceleration, evaluated separately.
- Alignment success rate no more than 5 percentage points lower.
- Median completion time no more than 10% longer.

Report peak acceleration as a complementary measure; do not describe motion as uniformly smoother if reduced RMS hides larger spikes. If baseline acceleration is effectively zero, a relative percentage is not meaningful; define an absolute threshold during validation instead.

Report uncertainty for the reductions and success-rate difference. Compare completion times on the episodes where both controllers succeed, and also report all-episode time-to-success with failures assigned the common timeout. This prevents faster completion on a smaller subset of easy successes from hiding a loss in capability.

Finally, test whether MPC improves the tradeoff relative to the tuned EMA baseline. Meeting the targets against the original controller is not evidence that prediction was necessary if ordinary smoothing achieves the same result.

Report transfer success degradation separately as the difference between the frozen controller's Elastica and MuJoCo success rates. The MPC smoothness targets do not define whether transfer itself succeeded. Establish a transfer criterion before held-out testing rather than assuming the simulators will yield equal performance.

## Risks and Interpretation

| Risk | Practical response |
| --- | --- |
| Adapter errors resemble policy-transfer failures | Validate prescribed motions, frames, observations, timing, and success rules first. |
| Nominal parameters produce different dynamics across simulators | Calibrate response traces; distinguish intended mismatch from mapping mistakes. |
| MPC has exact-state/model access while RL does not | Label oracle conditions and add an observation-based condition where feasible. |
| Full rod dynamics make planning too expensive | Benchmark previews, then consider a reduced predictor and report model-reduction error. |
| Tip history omits cable shape or velocities | Validate conditional errors; add a small set of features/history if necessary. |
| Model exploits unphysical direction predictions | Track direction-norm drift, shorten horizon, and restrict corrections. |
| Corrected rollouts leave the identification distribution | Log prediction errors and add a separately reported data-collection iteration if needed. |
| Short horizons suppress useful retreat or alignment maneuvers | Sweep horizons and correction weights on validation episodes. |
| Sensor noise dominates finite differences at 2 ms | Start noiseless, then use filtered estimates and a separately tested noise condition. |
| Contact invalidates the local model | Define the operating region and evaluate contact-rich insertion separately. |
| MPC acts mainly as a smoother | Use matched smoothing and zero-alignment ablations. |
| Solver latency is too high | Report it honestly; investigate acados or matched slower updates as later work. |

Constraint satisfaction in a predictive model is not a proof of cable safety or closed-loop stability in the plant. Finite-horizon geometric improvement is not a guarantee of better long-term return.

## Optional RMA Prediction Extension

Once nominal standalone MPC or the optional correction controller is useful, vary one hidden material property, initially stiffness or damping. Test whether a reduced predictor conditioned on the existing RMA context improves predictions and control:

$$
\hat y_{t+1}=f_\theta(y_t,y_{t-1},a_t,z_t).
$$

Keep the policy and RMA encoder frozen first. Train only the predictor. A policy-useful latent is not automatically prediction-useful; compare it against a fixed model and a model using history without the latent. Hold the latent fixed during each short imagined rollout and update it from real observations between solves.

Distinguish privileged phase-1 latents from history-derived phase-2 latents. A true-parameter model can serve as an explicitly labeled oracle diagnostic, not a deployable baseline. Do not assume an RMA latent can be inverted into physical parameters.

Suggested comparison within Elastica: frozen DRLR+RMA, standalone MPC with a fixed predictor, and standalone MPC with an RMA-conditioned predictor under held-out material values. If studying correction instead, compare fixed and adaptive correction layers with the same frozen controller. Add this only after the nominal experiment is complete.

## Deliverables and Completion Criteria

- A validated Elastica alignment environment adapter preserving the chosen checkpoint's contracts.
- Frozen-controller results in MuJoCo and Elastica, separating transfer from adapter/calibration effects.
- Standalone MPC results in Elastica with model/state access and calibration-data use documented.
- A matched controller comparison reporting success, completion time, gripper smoothness, cable oscillation, and latency.
- A documented data/action/coordinate contract and reproducible calibration traces, plus an identification dataset if fitting a reduced predictor.
- A validated prediction model or direct rod-preview interface with held-out prediction results.
- Tested reset and fallback behavior; an MPC correction evaluation path only if implementing the optional hybrid extension.
- Paired baseline/ablation results and latency measurements.
- Example trajectories showing a benefit, a failure, and a representative unchanged case.
- A short report stating whether the quantitative smoothness targets and performance guardrails were met, whether prediction helped beyond tuned EMA smoothing, where the model failed, and which assumptions limited the result.

The full case study is complete when it reproducibly characterizes frozen-controller transfer and compares standalone MPC with the transferred controller within Elastica. A negative result is useful if it distinguishes adapter/calibration issues, genuine transfer failure, insufficient prediction, an unsuitable objective/horizon, and a controller that already performs well. The optional correction layer is not required for completion.

## References

- [Fall School on Model Predictive Control and Reinforcement Learning](https://www.syscop.de/teaching/ws2026/fall-school-model-predictive-control-and-reinforcement-learning): project context and MPC/RL tooling.
- [CasADi Opti documentation](https://web.casadi.org/docs/#opti-stack): variables, parameters, constraints, and solving the prototype.
- [acados Python interface](https://docs.acados.org/python_interface/index.html): discrete dynamics and the later efficient-solver implementation.
- [PyElastica documentation](https://docs.cosseratrods.org/en/latest/index.html): Cosserat-rod environment and simulation interfaces.
- [PyElastica rod API](https://docs.cosseratrods.org/en/latest/api/rods.html): rod-state variables and material/geometry parameters.
- [PyElastica discretization guide](https://docs.cosseratrods.org/en/latest/guide/discretization.html): resolution, integration stability, convergence checks, and runtime considerations.
- Existing `dlo-manipulation-lab` implementation paths listed above: integration assumptions inspected for this plan.

This selected case-study plan supersedes the broad project recommendations in `project-suggestions.md` for the present discussion; it does not change that earlier file or implement the controller.
