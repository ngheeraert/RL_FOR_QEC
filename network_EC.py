# Policy network used by the RL agent (state-aware network in Fösel et al., PRX 2018).
# Maps an input representation of the quantum evolution to action probabilities pi(a|s).
# This network is the state-aware agent (two dense ReLU layers, softmax head).

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense 

# -----------------------------------------------------------------------------
# Feed-forward policy network pi_θ(a|s)
# - Input: state vector 'state' (constructed in quantum_simulator_EC.system.generate_net_input_state)
# - Output: probability distribution over discrete actions (gates/measurements)
# -----------------------------------------------------------------------------

class PolicyGradientNetwork(keras.Model):
    
    
    def __init__(self, n_actions, fc1_dims=300, fc2_dims=300):
        # n_actions: size of the discrete action set (idle, CNOTs, bit-flips, measurements)
        # fc*_dims: hidden-layer widths 

        super(PolicyGradientNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state):
        # Forward pass returning π(a|state).

        value = self.fc1(state)
        value = self.fc2(value)
        pi = self.pi(value)

        return pi
