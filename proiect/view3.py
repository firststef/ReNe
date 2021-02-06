import gym
from time import sleep
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

env = gym.make('BreakoutDeterministic-v4')
num_of_actions = env.action_space.n

render = True


class BreakoutNeuralNet(tf.keras.Model):
    def __init__(self, outs):
        super(BreakoutNeuralNet, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")
        self.conv_3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dense2 = tf.keras.layers.Dense(outs, dtype=tf.float32)

    def call(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flat(x)
        x = self.dense1(x)
        return self.dense2(x)


class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = []
        self.maxlen = maxlen
        self.is_over = False

    def add(self, q_state, q_action, q_reward, q_next_state):
        if self.is_over:
            self.buffer[np.random.randint(0, self.maxlen - 1)] = (q_state, q_action, q_reward, q_next_state)
        else:
            self.buffer.append((q_state, q_action, q_reward, q_next_state))
            if len(self.buffer) >= self.maxlen:
                self.is_over = True

    def get_all(self):
        states = [x[0] for x in self.buffer if x is not None]
        actions = [x[1] for x in self.buffer if x is not None]
        rewards = [x[2] for x in self.buffer if x is not None]
        nexts = [x[3] for x in self.buffer if x is not None]

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(nexts)

    def clear(self):
        self.buffer.clear()
        self.is_over = False


class CyclingBuffer:
    def __init__(self, maxlen):
        self.buffer = []
        self.maxlen = maxlen
        self.is_over = False

    def add(self, elem):
        if self.is_over:
            self.buffer = self.buffer[1:]
            self.buffer.append(elem)
        else:
            self.buffer.append(elem)
            if len(self.buffer) >= self.maxlen:
                self.is_over = True


# Hyper parameters
alpha = 0.1
gamma = 0.99
epsilon = 1
num_of_history_actions = 4
update_models = 10000

main_model = BreakoutNeuralNet(num_of_actions)
decision_model = BreakoutNeuralNet(num_of_actions)
decision_model.set_weights(main_model.get_weights())
replay_buffer = ReplayBuffer(150)
huber = keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam(1e-4)
states_c = CyclingBuffer(num_of_history_actions)
next_states_c = CyclingBuffer(num_of_history_actions)

# For plotting metrics
episode_reward_history = []
load = 'next9/largepic'


def actor_action(a_state):
    scores = main_model(a_state)
    choice = random.choices(range(num_of_actions))[0]
    return choice


for episode in range(0, 10000000):
    state = env.reset()
    env.step(1)

    reward, episode_reward = 0, 0
    previous_lives = 5
    done = False

    # Filling array at start
    b_state = state
    b_state = np.mean(b_state, 2, keepdims=False)
    b_state = b_state[35:195]
    for i in range(num_of_history_actions):
        states_c.add(b_state)
        next_states_c.add(b_state)

    b_state = np.array(states_c.buffer)
    b_state = b_state.reshape((160, 160, 4))
    b_state = np.asarray([b_state])
    main_model(b_state)
    decision_model(b_state)

    if load and episode == 0:
        model = keras.models.load_model(load)
        print([x.shape for x in model.get_weights()])
        state = np.array(states_c.buffer)
        state = state.reshape((160, 160, 4))
        main_model(np.asarray([state]))
        main_model.set_weights(model.get_weights())
        decision_model(np.asarray([state]))
        decision_model.set_weights(main_model.get_weights())
        continue

    while not done:
        # Make a decision
        state = np.mean(state, 2, keepdims=False)  # RGB to grayscale
        state = state[35:195]
        states_c.add(state)
        state = np.array(states_c.buffer)
        state = state.reshape((160, 160, 4))
        state = np.asarray([state])
        action = actor_action(state)

        # Execute the action and get the new state
        next_state, reward, done, info = env.step(action)

        if render:
            sleep(0.01)
            env.render()

        state = next_state