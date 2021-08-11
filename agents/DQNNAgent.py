from collections import deque
import random

import numpy as np

import tensorflow as tf

class DDQNAgent:
    def __init__(self, state_size, action_size, initial_state, get_illegal_actions, build_model):
        self.initial_state = initial_state
        self.state_size = state_size
        self.action_size = action_size
        self.get_illegal_actions = get_illegal_actions
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.5  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.learning_rate = 0.01
        self.online_model = build_model(initial_state, state_size, action_size)
        self.target_model = build_model(initial_state, state_size, action_size)
        self.update_weights()
        self.replay_counter = 1

    def remember(self, state, action, reward, next_state, done):
        #Function adds information to the memory about last action and its results
        self.memory.append((state, action, reward, next_state, done)) 

    def get_action(self, state):
        """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action.
            otherwise - the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """

        possible_actions = list(range(self.action_size))

        # agent parameters:
        epsilon = self.epsilon

        action = None

        if random.random() <= epsilon:
            action = random.choice(possible_actions)
            while action in self.get_illegal_actions(state):
                action = random.choice(possible_actions)
            # print(action, self.get_illegal_actions(state))
            return action
        
        return self.get_best_action(state)

  
    def get_best_action(self, state):
        """
        Compute the best action to take in a state.
        """

        possible_actions = list(range(self.action_size))
        if len(possible_actions) == 0:
            return None

        Q_values = self.online_model(tf.convert_to_tensor([state])).numpy()[0]
        Q_values[self.get_illegal_actions(state)] = float("-Inf")
        best_actions = np.reshape(np.argmax(Q_values), (-1))
        return random.choice(best_actions)

    def replay(self, batch_size):
        """
        Function learn network using randomly selected actions from the memory. 
        First calculates Q value for the next state and choose action with the biggest value.
        Target value is calculated according to:
                Q(s,a) := (r + gamma * max_a(Q(s', a)))
        except the situation when the next action is the last action, in such case Q(s, a) := r.
        In order to change only those weights responsible for choosing given action, the rest values should be those
        returned by the network for state state.
        The network should be trained on batch_size samples.
        After each 10 Q Network trainings parameters should be copied to the target Q Network
        """
        batch = random.sample(self.memory, k=batch_size)
        states, actions, rewards, next_states, dones = np.array(batch, dtype=object).transpose()
        actions = np.asarray(actions).astype(np.int)
        dones = np.asarray(dones).astype(np.bool)
        rewards = np.asarray(rewards).astype(np.float64)
        # Get rid of nested numpy arrays
        next_states_as_tf = tf.convert_to_tensor([list(x) for x in next_states])
        Q_ns = self.target_model.predict(next_states_as_tf, batch_size=batch_size)
        best_Q_ns = np.max(Q_ns, axis=1)
        targets = self.online_model.predict(
                tf.convert_to_tensor([list(x) for x in states]),
                batch_size=batch_size)
        targets[np.arange(targets.shape[0]), actions] =\
            (1 - dones) * (rewards + self.gamma * best_Q_ns) +\
            dones * rewards
        
        x = tf.convert_to_tensor([states])[0]
        y = tf.convert_to_tensor(targets)
        x = np.reshape(x, (batch_size, -1))
        y = np.reshape(y, (batch_size, -1))
        self.online_model.fit(x=x, y=y, batch_size=batch_size, verbose=0)
        
        self.replay_counter += 1
        if self.replay_counter == 10:
            self.replay_counter = 1
            self.update_weights()

    def update_epsilon_value(self):
        #Every each epoch epsilon value should be updated according to equation: 
        #self.epsilon *= self.epsilon_decay, but the updated value shouldn't be lower then epsilon_min value
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        
    def update_weights(self):
        """copy trained Q Network params to target Q Network"""
        self.target_model.set_weights(self.online_model.get_weights())
        
    def turn_off_learning(self):
        self.epsilon = 0
