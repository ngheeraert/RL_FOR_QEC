# Policy-gradient RL agent for the state-aware network (Fösel et al., PRX 2018).
# 
# This implements a REINFORCE-style update with:
#   - discounted returns G (per time step) and a running baseline b_t to reduce variance,
#   - entropy regularization to encourage exploration,
#   - optional natural-gradient preconditioning using a diagonal Fisher-information estimate.

import tensorflow as tf
from network_EC import PolicyGradientNetwork
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys
from copy import copy, deepcopy

def _to_np(x, dtype):
    # Convert TF tensors (including on metal device) to host numpy first
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    return np.asarray(x, dtype=dtype)

# -----------------------------------------------------------------------------
# Agent: stores trajectories and updates the policy network parameters θ
# -----------------------------------------------------------------------------

class Agent:
    
    def __init__(self, alpha=0.0001/np.sqrt(10), gamma=0.95, n_actions=4,
                 layer1_size=300, layer2_size=300, lambda_entr=5e-3):

        # --- Hyperparameters / coefficients (paper terminology) ---
        # gamma: discount factor for future rewards.
        # lambda_pol: scales the policy-gradient term.
        # lambda_entr: scales entropy regularization (encourages exploration).
        # kappa: smoothing factor for the running baseline b_t.


        self.kappa = 0.9
        self.lambda_pol = 4.0
        self.lambda_entr = lambda_entr
        self.gamma = gamma
        self.n_actions = n_actions
        # Clear trajectory buffers after the update.
        self.state_memory = []
        self.action_memory = []
        self.reward1_memory = []
        self.reward2_memory = []
        self.mean_returns = []
        # Policy network πθ(a|s): maps environment state representation -> action probabilities
        self.policy = PolicyGradientNetwork(n_actions, layer1_size, layer2_size)
        # Optimizer for θ (Adam is used here; policy-gradient itself comes from the loss below)
        self.policy.compile(optimizer=Adam(learning_rate=alpha, beta_1=0.9, beta_2=0.999))
        self.number_of_epochs_trained = 0
        # Natural gradient (diagonal Fisher) settings
        # --- Natural policy gradient option ---
        # Fösel et al. mention using a natural policy-gradient update (Appendix H).
        # Here we approximate the Fisher information matrix by its diagonal, estimated from
        # score-function gradients, and use it to precondition the gradient.
        self.use_natural_gradient = True
        self.fisher_beta = 0.9
        self.fisher_damping = 1e-3
        self._fisher_diag = None  # lazily initialized to match variables
        
    def load_policy(self, filename):
        self.policy = tf.keras.models.load_model(filename)

    def choose_action(self, state):
        # Sample an action a ~ πθ(a|s) for a single state vector.
        # Sampling (rather than argmax) is essential during training for exploration.

        state_tf = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.policy(state_tf)  # shape (1, n_actions)
        # Sample without TFP (avoid log(0))
        probs = tf.clip_by_value(probs, 1e-8, 1.0)
        logits = tf.math.log(probs)
        action = tf.random.categorical(logits, 1)
        return int(action.numpy()[0, 0])

    def choose_actions(self, states_batch):
        # states_batch: (B, obs_dim) numpy array
        states_tf = tf.convert_to_tensor(states_batch, dtype=tf.float32)
        probs = self.policy(states_tf)  # (B, n_actions)
        probs = tf.clip_by_value(probs, 1e-8, 1.0)
        logits = tf.math.log(probs)
        actions = tf.random.categorical(logits, 1)[:, 0]
        return actions.numpy().astype(np.int32)

    def choose_action_batch(self, state):
        # Identical to choose_action (kept as a separate method for compatibility with notebooks).

        state_tf = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.policy(state_tf)  # shape (1, n_actions)
        # Sample without TFP (avoid log(0))
        probs = tf.clip_by_value(probs, 1e-8, 1.0)
        logits = tf.math.log(probs)
        action = tf.random.categorical(logits, 1)
        return int(action.numpy()[0, 0])

    def store_transition(self, observation, action, reward1, reward2):
        # Store a single transition (one time step). In this project we typically store
        # full trajectories (states/actions/rewards over N_gates steps) and then call learn().

        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward1_memory.append(reward1)
        self.reward2_memory.append(reward2)
        
    def store_batch(self, states, actions, rewards1, rewards2):
        # Store a full batch of trajectories.
        # states:  [batch, N_gates, obs_dim]
        # actions: [batch, N_gates]
        # rewards*: [batch, N_gates]

        #-- OLD CODE
        #self.state_memory = states
        #self.action_memory = actions
        #self.reward1_memory = rewards1
        #self.reward2_memory = rewards2

        self.state_memory  = _to_np(states,  np.float32)
        self.action_memory = _to_np(actions, np.int32)
        self.reward1_memory = _to_np(rewards1, np.float32)
        self.reward2_memory = _to_np(rewards2, np.float32)


    def learn(self):
        # ---------------------------------------------------------------------
        # Policy update (REINFORCE / policy gradient)
        #
        # The environment provides two reward streams (reward1, reward2) per step:
        #   - reward1: shaped reward derived from recoverable quantum information (RQ)
        #   - reward2: sparse penalty term (e.g. when RQ collapses)
        #
        # We build per-time-step returns G[i,t] and subtract a running baseline b[t]
        # to form an advantage signal. The loss implements
        #   L = - E[ sum_t (G-b)_t * log πθ(a_t|s_t)  +  entropy bonus ]
        # ---------------------------------------------------------------------

        actions = np.array(self.action_memory)
        rewards1 = np.array(self.reward1_memory)
        rewards2 = np.array(self.reward2_memory)
        #actions = np.asarray(self.action_memory, dtype=np.int32)
        #rewards1 = np.asarray(self.reward1_memory, dtype=np.float32)
        #rewards2 = np.asarray(self.reward2_memory, dtype=np.float32)
        
        batch_size = np.shape(rewards1)[0]
        N_gates = np.shape(rewards1)[1]
        current_epoch = len(self.mean_returns)

        # Compute discounted returns G for every (trajectory i, time t)
        # Note: rewards1 is discounted and normalized by (1-gamma) as in some
        # continuing-task formulations; rewards2 is added as an extra immediate term.

        #-- OLD CODE
        G = np.zeros_like(rewards1)
        #for i in range(batch_size):
        #    
        #    for t in range( N_gates ):
        #        G_sum = 0
        #        for k in range(N_gates-t):
        #            G_sum += rewards1[i,t+k] * self.gamma**k
        #        G[i,t] = (1-self.gamma)*G_sum + rewards2[i,t]

        #-- NEW CODE
        # rewards1, rewards2: (B, T)
        B, T = rewards1.shape
        G = np.zeros_like(rewards1, dtype=np.float32)

        # discounted cumulative sum in reverse
        running = np.zeros((B,), dtype=np.float32)
        for t in reversed(range(T)):
            running = rewards1[:, t] + self.gamma * running
            G[:, t] = (1.0 - self.gamma) * running + rewards2[:, t]
        
        self.mean_returns.append( G[:,:].mean(axis=0) )
        
        # Running baseline b[t] (exponential smoothing over past mean returns)
        # This reduces the variance of the policy-gradient estimate.
        b = np.ones( N_gates )
        b *= (1-self.kappa)
        
        for t in range(N_gates):
            factor = 0
            if (current_epoch>0): 
                for n in range(current_epoch):
                    factor += self.kappa**n * self.mean_returns[current_epoch-1-n][t]
                b[t] *= factor
                
        #G_minus_b = tf.convert_to_tensor(G - b[None, :], dtype=tf.float32)
        G_minus_b = tf.convert_to_tensor( G - b )

        #-- NEW CODE W OPTIMIZATION
        with tf.GradientTape(persistent=True) as tape:
            # states: (B, T, obs_dim), actions: (B, T), advantages: (B, T)
            states_tf  = tf.convert_to_tensor(self.state_memory, dtype=tf.float32)
            actions_tf = tf.convert_to_tensor(self.action_memory, dtype=tf.int32)
            adv_tf     = tf.cast(G_minus_b, tf.float32)  # already tensor (B, T)

            B = tf.shape(states_tf)[0]
            T = tf.shape(states_tf)[1]

            # Flatten trajectories so we do ONE network call:
            # (B*T, obs_dim)
            flat_states  = tf.reshape(states_tf, (-1, tf.shape(states_tf)[2]))
            flat_actions = tf.reshape(actions_tf, (-1,))
            flat_adv     = tf.reshape(adv_tf, (-1,))

            probs = self.policy(flat_states, training=True)                 # (B*T, n_actions)
            probs = tf.clip_by_value(probs, 1e-8, 1.0)
            log_probs = tf.math.log(probs)

            # pick log π(a_t|s_t) for all steps at once
            idx = tf.stack([tf.range(tf.shape(flat_actions)[0]), flat_actions], axis=1)
            log_probs_a = tf.gather_nd(log_probs, idx)                      # (B*T,)

            # entropy term: -sum_a p log p  (note probs*log_probs <= 0)
            entropy_term = tf.reduce_sum(probs * log_probs) / tf.cast(B * T, tf.float32)

            # policy gradient term
            pg_term = tf.reduce_sum(flat_adv * log_probs_a) / tf.cast(B, tf.float32)

            loss = -(self.lambda_pol * pg_term - self.lambda_entr * entropy_term)

            # score for Fisher estimate (mean log-prob of taken action)
            score = tf.reduce_mean(log_probs_a)
        

        #-- OLD CODE
        # Build the policy-gradient loss over the batch.
        # We also build a 'score' (mean log πθ(a_t|s_t)) to estimate the diagonal Fisher.
        #with tf.GradientTape(persistent=True) as tape:
        #    loss = 0.0
        #    score_sum = 0.0  # used to estimate diagonal Fisher from score function grads
        #    for i in range(batch_size):

        #        states_tf = tf.convert_to_tensor(self.state_memory[i], dtype=tf.float32)
        #        actions_tf = tf.convert_to_tensor(self.action_memory[i], dtype=tf.int32)

        #        probs = self.policy(states_tf, training=True)
        #        # Numerical stability: avoid log(0) and NaNs in entropy
        #        probs = tf.clip_by_value(probs, 1e-8, 1.0)
        #        log_probs = tf.math.log(probs)

        #        slice_indices = tf.transpose(tf.stack((tf.range(0, N_gates), actions_tf)))
        #        log_probs_a = tf.gather_nd(log_probs, slice_indices)

        #        # Entropy regularization term (encourages broader action distributions)
        #        # Entropy term: -sum_a p log p  (note: probs*log_probs is <= 0)
        #        probs_log_probs = probs * log_probs 
        #        sum_over_s_and_a = tf.reduce_sum(probs_log_probs)

        #        # Advantage estimate A_t = (G - b)_t (paper: baseline-subtracted return)
        #        # Use advantage (G - b) as in the paper
        #        adv = tf.cast(G_minus_b[i, :], tf.float32)

        #        loss += -(
        #            self.lambda_pol * tf.reduce_sum(adv * log_probs_a)
        #            - self.lambda_entr * sum_over_s_and_a / N_gates
        #        )

        #        # Score-function term used for Fisher estimate (natural gradient)
        #        # Score term for Fisher estimate (no advantage scaling)
        #        score_sum += tf.reduce_sum(log_probs_a)

        #    loss /= batch_size
        #    score = score_sum / tf.cast(batch_size * N_gates, tf.float32)

        ## Compute gradients for the policy loss, and (separately) score gradients.
        grads = tape.gradient(loss, self.policy.trainable_variables)
        score_grads = tape.gradient(score, self.policy.trainable_variables)
        del tape

        # Diagonal Fisher-information estimate F ≈ E[ (∂/∂θ log π)^2 ]
        # Updated using an exponential moving average (fisher_beta).
        # Diagonal Fisher estimate (EMA of squared score gradients)
        if self._fisher_diag is None:
            self._fisher_diag = [tf.ones_like(v, dtype=tf.float32) for v in self.policy.trainable_variables]

        new_fisher = []
        for f_old, g_s in zip(self._fisher_diag, score_grads):
            if g_s is None:
                new_fisher.append(f_old)
                continue
            g_s = tf.cast(g_s, tf.float32)
            f_est = tf.square(g_s)
            f_new = self.fisher_beta * f_old + (1.0 - self.fisher_beta) * f_est
            new_fisher.append(f_new)
        self._fisher_diag = new_fisher

        # Natural-gradient preconditioning: g_nat = g / (F + damping)
        # (Diagonal approximation; damping ensures numerical stability.)
        if self.use_natural_gradient:
            nat_grads = []
            for g, f in zip(grads, self._fisher_diag):
                if g is None:
                    nat_grads.append(None)
                    continue
                nat_grads.append(tf.cast(g, tf.float32) / (tf.cast(f, tf.float32) + self.fisher_damping))
            grads_to_apply = nat_grads
        else:
            grads_to_apply = grads

        # Apply (possibly preconditioned) gradients with Adam.
        self.policy.optimizer.apply_gradients(zip(grads_to_apply, self.policy.trainable_variables))


        self.state_memory = []
        self.action_memory = []
        self.reward1_memory = []
        self.reward2_memory = []
        self.number_of_epochs_trained += 1
        
        
