# Reinforcement Learning for Quantum-Memory Protection (4-qubit simulator)

This codebase implements a compact research prototype for **learning feedback-control policies** that protect quantum information in a noisy multi-qubit device.

At a high level:
- a **quantum simulator** models open-system dynamics, measurements, and a discrete set of control operations,
- a **policy network** maps a compact representation of the current quantum-channel state to an action distribution,
- a **policy-gradient agent** samples actions, collects trajectories, and updates the policy to maximize long-term preservation of quantum information.

(Reference: Fösel et al., *Reinforcement Learning with Neural Networks for Quantum Feedback*, Phys. Rev. X 8, 031084 (2018).)

---

## Repository contents

### `quantum_simulator_EC.py` — environment / simulator
Implements the RL “environment” for a **4-qubit quantum memory** subject to stochastic noise and measurement back-action.

Key responsibilities:
- Defines the **action space** (idle, two-qubit entangling gates, single-qubit flips, single-qubit measurements).
- Evolves the system forward in time under noise.
- Applies unitary gates and measurement updates (including branching / stochastic outcomes).
- Computes an information-preservation score and converts it into rewards.

### `network_EC.py` — policy model
Defines a feed-forward Keras model that outputs **action probabilities** (softmax). The default architecture is a two-hidden-layer MLP sized for the state representation produced by the simulator.

### `reinforce_keras_EC.py` — policy-gradient agent
Defines the training logic:
- Samples actions from the policy distribution.
- Stores per-trajectory `(state, action, reward)` sequences.
- Computes discounted returns and an optional baseline.
- Updates the policy using **REINFORCE-style** gradients with **entropy regularization**.
- Optionally applies a lightweight diagonal-Fisher preconditioning (a practical “natural-gradient-like” step).

---

## State representation (what the network “sees”)

The simulator maintains four density matrices that together encode how a single logical-qubit state is transformed by the noisy dynamics and control operations. At each time step, these matrices are compressed into a fixed-length vector by:
- extracting a limited number of dominant components (PCA-like truncation),
- concatenating real and imaginary parts,
- appending a few auxiliary features (e.g., flags related to measurement informativeness),
- appending the previous action index.

This produces a consistent input vector suitable for an MLP policy.

---

## Action space (what the agent can do)

For 4 qubits, the discrete control set includes:
- **idle** (do nothing),
- **two-qubit entangling operations** between ordered qubit pairs,
- **single-qubit bit flips**,
- **single-qubit measurements** in the Z basis.

The simulator exposes these actions through its `actions` list, and the policy outputs a probability for each action index.

---

## Objective and rewards (what is being optimized)

The training objective is to learn policies that **preserve quantum information** over time.

Each step yields rewards derived from an information-preservation score computed from the channel representation (the four density matrices). Intuitively:
- policies are rewarded for keeping the memory “recoverable” (high information score),
- policies are penalized when information is irreversibly destroyed.

The agent maximizes the expected discounted sum of these rewards.

---

## Quickstart

This repo is often driven from a notebook, but the flow is:

1. Create the simulator (`system`).
2. Create the agent (`Agent`), with `n_actions = len(s.actions)`.
3. Repeat:
   - roll out trajectories by alternating **state → action → reward → evolution**,
   - store trajectories,
   - call `agent.learn()` to update the policy.

---

## Notes on saving/loading

If your policy model is subclassed, ensure its variables are created (e.g., by calling it once on a dummy input) before loading weights.
