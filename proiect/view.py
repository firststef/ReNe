from datetime import datetime

import gym
from time import sleep
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow import keras

env = gym.make('Breakout-ram-v0')
num_of_actions = env.action_space.n

render = True


class BreakoutNeuralNet(tf.keras.Model):
    def __init__(self, outs):
        super(BreakoutNeuralNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(outs, dtype=tf.float32)  # No activation

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


main_model = BreakoutNeuralNet(num_of_actions)
decision_model = BreakoutNeuralNet(num_of_actions)
decision_model.compile(optimizer='adam', loss='mse')
decision_model.set_weights(main_model.get_weights())
mse = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(1e-4)

# Hyper parameters
alpha = 0.1
gamma = 0.99
epsilon = 1

# For plotting metrics
episode_reward_history = []
#backup
#load = 'breakout2020-12-31_18_58_43_193743' doar apasa fire, nimic altceva, cu actiune mica 0.1
#load = 'backup_breakout2020-12-31_06_38_34_631078.pickle' sta cam mereu in dreapta mai mult asteapta sa se loveasca mingea de el - are reward *3 si +1 per life
#load = 'backu_breakout2020-12-31_06_38_34_631078.pickle' # much better - vechi
#load = 'breakout2020-12-31_18_53_05_012586' nici macar fire nu apasa, cred ca aici ii dadeam reward ca apasa fire prea mare
#load = 'breakout2020-12-31_11_50_12_351150' sta numai in stanga nu face nimica
#next1
#load = 'backup_breakout2020-12-31_06_38_34_631078.pickle' best so far i think it learnt to press the button a little and a little of play, this is reward*3
#load = 'breakout2020-12-31_18_58_43_193743' doesnt do much but sometimes presses fire, this is the one with +0.1 per fire
#load = 'breakout2020-12-31_18_58_43_193743' great doesnt do anything
#load = 'backup_breakout2020-12-31_06_38_34_631078.pickle'# fkin great now only presses fire
#load = 'next1/backup_breakout2020-12-31_06_38_34_631078.pickle'
#load = 'next2/backup_breakout2020-12-31_06_38_34_631078.pickle'  # not impressive
#load = 'next2/simple_min1_at_end' # stoica moment
#load = 'next3/backup_breakout2020-12-31_06_38_34_631078.pickle'
#load = 'next3/simple_min1_at_end'  # pare best
#load = 'next3/breakout2020-12-31_18_58_43_193743'
#load = 'next4/all_backp' doesnt do anything good, just fidges around
#load = 'next4/backup_breakout2020-12-31_06_38_34_631078.pickle'
#load = 'next5/learnratesimple'
#load = 'next5/learnrate' for this i tried learn_rate = 0.01 and 0.1 per fire
#load = 'next5/back3_simple' somehow stays in place and gets points
#load = 'next5/learnratebig' ok
#load = 'next6/learnratebig' # sta numai in stanga
#load = 'next7/donesbatch'
#load = 'next8/fixddqnmany'  #fidget
#load = 'next8/fixedqqnegative'  #stoic
#load = 'next8/anotherfixedqq'  #nimic = alpha 0.5, huber
#load = 'next8/donesbatch' #nimic
#load = 'next8/donessnoalpha' sta in stanga
# load = None


def actor_action(a_state):
    scores = main_model(a_state)
    choice = np.argmax(scores)
    return choice


state = env.reset()

done = False

model = keras.models.load_model(load)
print([x.shape for x in model.get_weights()])
main_model(np.asarray([state]))
main_model.set_weights(model.get_weights())
decision_model(np.asarray([state]))
decision_model.set_weights(main_model.get_weights())

for i in range(1000):
    state = env.reset()
    done = False

    counter = 0

    while not done:
        counter = counter + 1
        # Make a decision
        state = np.asarray([state])
        action = actor_action(state)
        if random.uniform(0, 1) <= 0.001:
            action = 1

        # Execute the action and get the new state
        next_state, reward, done, info = env.step(action)
        # print(reward)
        if counter > 600:
            break

        if render:
            sleep(0.01)
            env.render()

        state = next_state
