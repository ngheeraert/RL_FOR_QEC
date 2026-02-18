import tensorflow as tf
from network_EC import PolicyGradientNetwork
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys
from copy import copy, deepcopy

class Agent:
    
    def __init__(self, alpha=0.0001/np.sqrt(10), gamma=0.95, n_actions=4,
                 layer1_size=300, layer2_size=300, lambda_entr=5e-3):

        self.kappa = 0.9
        self.lambda_pol = 4.0
        self.lambda_entr = lambda_entr
        self.gamma = gamma
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward1_memory = []
        self.reward2_memory = []
        self.mean_returns = []
        self.policy = PolicyGradientNetwork(n_actions, layer1_size, layer2_size)
        self.policy.compile(optimizer=Adam(learning_rate=alpha, beta_1=0.9, beta_2=0.999))
        self.number_of_epochs_trained = 0
        # Natural gradient (diagonal Fisher) settings
        self.use_natural_gradient = True
        self.fisher_beta = 0.9
        self.fisher_damping = 1e-3
        self._fisher_diag = None  # lazily initialized to match variables
        
    def load_policy(self, filename):
        self.policy = tf.keras.models.load_model(filename)

    def choose_action(self, state):
        state_tf = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.policy(state_tf)  # shape (1, n_actions)
        # Sample without TFP (avoid log(0))
        probs = tf.clip_by_value(probs, 1e-8, 1.0)
        logits = tf.math.log(probs)
        action = tf.random.categorical(logits, 1)
        return int(action.numpy()[0, 0])

    def choose_action_batch(self, state):
        state_tf = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.policy(state_tf)  # shape (1, n_actions)
        # Sample without TFP (avoid log(0))
        probs = tf.clip_by_value(probs, 1e-8, 1.0)
        logits = tf.math.log(probs)
        action = tf.random.categorical(logits, 1)
        return int(action.numpy()[0, 0])

    def store_transition(self, observation, action, reward1, reward2):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward1_memory.append(reward1)
        self.reward2_memory.append(reward2)
        
    def store_batch(self, states, actions, rewards1, rewards2):
        self.state_memory = states
        self.action_memory = actions
        self.reward1_memory = rewards1
        self.reward2_memory = rewards2

    def learn(self):
        actions = np.array(self.action_memory)
        rewards1 = np.array(self.reward1_memory)
        rewards2 = np.array(self.reward2_memory)
        
        batch_size = np.shape(rewards1)[0]
        N_gates = np.shape(rewards1)[1]
        current_epoch = len(self.mean_returns)

        G = np.zeros_like(rewards1)
        for i in range(batch_size):
            
            for t in range( N_gates ):
                G_sum = 0
                for k in range(N_gates-t):
                    G_sum += rewards1[i,t+k] * self.gamma**k
                G[i,t] = (1-self.gamma)*G_sum + rewards2[i,t]
        
        self.mean_returns.append( G[:,:].mean(axis=0) )
        
        b = np.ones( N_gates )
        b *= (1-self.kappa)
        
        for t in range(N_gates):
            factor = 0
            if (current_epoch>0): 
                for n in range(current_epoch):
                    factor += self.kappa**n * self.mean_returns[current_epoch-1-n][t]
                b[t] *= factor
                
        G_minus_b = tf.convert_to_tensor( G - b )
        
        with tf.GradientTape(persistent=True) as tape:
            loss = 0.0
            score_sum = 0.0  # used to estimate diagonal Fisher from score function grads
            for i in range(batch_size):

                states_tf = tf.convert_to_tensor(self.state_memory[i], dtype=tf.float32)
                actions_tf = tf.convert_to_tensor(self.action_memory[i], dtype=tf.int32)

                probs = self.policy(states_tf, training=True)
                # Numerical stability: avoid log(0) and NaNs in entropy
                probs = tf.clip_by_value(probs, 1e-8, 1.0)
                log_probs = tf.math.log(probs)

                slice_indices = tf.transpose(tf.stack((tf.range(0, N_gates), actions_tf)))
                log_probs_a = tf.gather_nd(log_probs, slice_indices)

                # Entropy term: -sum_a p log p  (note: probs*log_probs is <= 0)
                probs_log_probs = probs * log_probs
                sum_over_s_and_a = tf.reduce_sum(probs_log_probs)

                # Use advantage (G - b) as in the paper
                adv = tf.cast(G_minus_b[i, :], tf.float32)

                loss += -(
                    self.lambda_pol * tf.reduce_sum(adv * log_probs_a)
                    - self.lambda_entr * sum_over_s_and_a / N_gates
                )

                # Score term for Fisher estimate (no advantage scaling)
                score_sum += tf.reduce_sum(log_probs_a)

            loss /= batch_size
            score = score_sum / tf.cast(batch_size * N_gates, tf.float32)

        grads = tape.gradient(loss, self.policy.trainable_variables)
        score_grads = tape.gradient(score, self.policy.trainable_variables)
        del tape

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

        self.policy.optimizer.apply_gradients(zip(grads_to_apply, self.policy.trainable_variables))


        self.state_memory = []
        self.action_memory = []
        self.reward1_memory = []
        self.reward2_memory = []
        self.number_of_epochs_trained += 1
        
        
