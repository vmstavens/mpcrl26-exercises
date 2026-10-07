# Project Suggestions: DRLR(DP)+RMA, DLO Manipulation, and MPCRL

These suggestions connect the [Fall School on Model Predictive Control and Reinforcement Learning](https://www.syscop.de/teaching/ws2026/fall-school-model-predictive-control-and-reinforcement-learning) curriculum to research on DRLR(DP)+RMA and deformable linear object (DLO) manipulation.

The working assumption is that DRLR means Deep Reinforcement Learning with Reference policy, DP is the diffusion reference policy, and RMA infers a dynamics representation from interaction history. The exact architecture and DLO task still need to be specified when selecting a project.

The strongest school project is predictive action selection for DRLR(DP)+RMA. The strongest longer-term direction is using RMA to adapt the dynamics model inside MPC.

## Curriculum Connections

The school timetable includes nonlinear MPC with acados, differentiable MPC, MPC-SAC, imitation learning, safety and stability, and a diffusion tutorial.

| Project | Main research question | School scope |
| --- | --- | --- |
| Predictive action selection | Can short rollouts improve selection between reference and RL actions? | Best starting point |
| RMA-conditioned MPC | Is adaptation more useful in the policy, predictive model, or both? | Feasible with an existing simulator |
| Adaptive predictive safety filter | Can adaptation reduce unnecessary intervention while controlling violations? | Strong demonstrator |
| RL learns MPC parameters | Can RL improve a constrained controller with fewer interactions? | Closest to MPC-SAC exercises |

## 1. Predictive Action Selection for DRLR(DP)+RMA

### Idea

Extend the existing action-selection mechanism with a short predictive rollout. Generate candidates from the diffusion reference policy and SAC policy, predict their consequences using a model conditioned on the RMA latent, and score them using accumulated reward plus a terminal value estimate. Execute the first action and repeat.

### Hypothesis

Under unfamiliar cable stiffness or friction, predictive evaluation selects better actions than critic estimates alone. This directly extends DRLR's concern with unreliable value estimates during exploration.

### School Prototype

Use one cable-shaping task, a few candidate sequences, and a short horizon. Keep candidate horizons and execution lengths matched.

Compare:

- The current action selector.
- Predictive selection with a fixed model.
- Predictive selection with an adaptation-conditioned model.

### Evaluation

Measure task success, interactions needed to reach a target performance, and how often the selector chooses an action whose realized return is worse than the alternative. In simulation, evaluate alternatives from matched initial states to make the selection comparison meaningful.

### Research Positioning

Related work already ranks and refines diffusion proposals using world models. The contribution should center on DRLR's exploration mechanism and adaptation under DLO dynamics changes.

Sources:

- [DRLR: Solving robotics tasks with prior demonstration via exploration-efficient deep reinforcement learning](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1682200/full)
- [Generative Predictive Control](https://computationalrobotics.seas.harvard.edu/GPC/)

## 2. Use RMA to Adapt an MPC Dynamics Model

### Idea

Feed the history-derived latent into a predictive model:

$$
z_t = h(\text{observation/action history}), \qquad
\hat{x}_{t+1} = f_\theta(x_t, u_t, z_t).
$$

MPC then plans using the currently inferred dynamics. The state could contain cable keypoints and gripper states.

### Hypothesis

A shared adaptive model enables better transfer across materials and goals than adapting only the policy.

### School Prototype

Initially vary just stiffness and damping in the existing simulator.

Compare:

- Adaptation in the policy only.
- Adaptation in the predictive model only.
- Adaptation in both the policy and predictive model.
- A fixed-model controller.
- An oracle supplied with true simulation parameters.

### Evaluation

Measure task success, prediction error, and recovery after changes in dynamics. Test held-out parameter combinations and goals.

### Research Positioning

A latent that helps the policy is not automatically useful for prediction. Test the existing latent first, then add a dynamics-prediction objective if necessary.

RMA-style adaptation for deformable manipulation already exists. The distinctive question is where adaptation belongs in a predictive control architecture.

Source: [RAPiD: Rapid Adaptation of Particle Dynamics for Generalized Deformable Object Mobile Manipulation](https://arxiv.org/abs/2603.18246)

## 3. An Adaptive Predictive Safety Filter

### Idea

Let DRLR(DP)+RMA propose an action sequence. Solve a small optimization problem that minimally changes it while respecting predicted cable clearance, stretch limits, and gripper motion bounds.

### Hypothesis

Adapting the model reduces both constraint violations and unnecessary corrections compared with a fixed conservative filter.

### School Prototype

Start with cable shaping around one obstacle. Check the whole cable, not just gripper clearance.

Compare:

- The original policy.
- A fixed-model filter.
- An adaptive filter.

### Evaluation

Report task success, constraint violations, intervention magnitude, and control latency.

### Research Positioning

Learning-based MPC with a safety filter has already been demonstrated for DLO manipulation. The extension would investigate how adaptation changes the performance-constraint tradeoff.

With an approximate learned model, distinguish observed constraint satisfaction from a formal safety guarantee.

Source: [Learning-Based MPC With Safety Filter for Constrained Deformable Linear Object Manipulation](https://ieeexplore.ieee.org/document/10423099/)

## 4. Let RL Learn MPC Parameters

### Idea

Build a compact MPC controller, then let SAC choose bounded parameters such as shape-tracking weights, action penalties, or reference offsets. The diffusion policy could provide the nominal reference, and RMA could condition the parameter choices.

### Hypothesis

Optimizing a small set of controller parameters needs fewer interactions and generalizes better than learning unrestricted corrections.

### School Prototype

Learn two or three parameters and keep constraint limits fixed.

Compare under the same interaction budget:

- Hand-tuned MPC.
- MPC with learned parameters.
- The existing DRLR baseline.

### Evaluation

Measure sample efficiency, task success, constraint violations, and transfer to unfamiliar dynamics.

### Curriculum Connection

This project fits the school's differentiable MPC and MPC-SAC exercises. leap-c exposes an acados-based PyTorch layer and solution sensitivities.

Source: [leap-c](https://github.com/leap-c/leap-c)

## Recommended School Project

**Research question:** Does adaptation-conditioned predictive action selection improve exploration efficiency in DLO manipulation under material changes?

Reuse the existing task and checkpoints. Start with cable keypoints rather than adding perception work, and reserve continuous MPC refinement for an extension.

The minimum useful result is a comparison of the current selector, a fixed-model predictive selector, and an adaptation-conditioned predictive selector under matched interaction budgets and material changes. This experiment would directly inform whether adapting the dynamics model inside MPC is worth pursuing as a larger research project.
