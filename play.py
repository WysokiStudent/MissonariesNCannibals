#!/usr/bin/env python
from matplotlib import pyplot as plt
import pickle
import numpy as np

import pygame
from agents.QLearningAgent import QLearningAgent
from agents.DQNNAgent import DDQNAgent
from game.missionariesandcannibals import MnC

import keras
from keras import Sequential, Input
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import MSE

# Possible algorithms are, 'Q-learning', 'DQNN'
# ALGORITHM = 'DQNN'
# DQNN_LEARN = True
ALGORITHM = 'Q-learning'

env = MnC()
env.reset()

if ALGORITHM == 'Q-learning':
    print("Q-learning")
    def play_and_train(env, agent):
        """
        This function should
        - run a full game, actions given by agent's e-greedy policy
        - train agent using agent.update(...) whenever it is possible
        - return total reward
        """
        total_reward = 0.0
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.update(state, action, reward, next_state)    
            state = next_state
            if done:
                break;
        return total_reward

    agent = QLearningAgent(alpha=0.01, epsilon=0.1, discount=0.99,
                           get_legal_actions=env.get_possible_actions)
    env.reset()
    env.turn_off_display()
    filtered_score_for_plotting = []
    filtered_score = -500
    theoretical_maximum = 500
    total_iterations = 0
    while filtered_score < theoretical_maximum:
        iteration_counter = 0
        while iteration_counter < 10:
            iteration_counter += 1
            score = play_and_train(env, agent)
            filtered_score = 99/100 * filtered_score + 1/100 * score
            print(theoretical_maximum, filtered_score)
            filtered_score_for_plotting.append(filtered_score)
        theoretical_maximum -= 1
        total_iterations += iteration_counter
    print("Stopped training with score ", theoretical_maximum, " after ",
            total_iterations, " iterations.")

    plt.plot(filtered_score_for_plotting)
    plt.ylabel("Filtered score")
    plt.xlabel("Epoch")
    plt.show()

    env.turn_on_display()
    state = env.reset()
    agent.turn_off_learning()
    done = False

    while not done:
        score = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        action = agent.get_action(state)
        next_state, reward, done, reward = env.step(action)
        score += reward
        state = next_state
        print("Score:", reward)
elif ALGORITHM == 'DQNN':
    print("DQNN")

    translate = {}
    translate[0] = 'm'
    translate[1] = 'mm'
    translate[2] = 'c'
    translate[3] = 'cc'
    translate[4] = 'cm'
    translate['m'] = 0
    translate['mm'] = 1
    translate['c'] = 2
    translate['cc'] = 3
    translate['cm'] = 4

    def translate_state(state: str) -> np.array:
        new_state = [0] * 13
        # 0 - strona po której jest łódka
        # 1 - misjonarz po lewej stronie
        # 2 - dwóch misjonarzy po lewej stronie
        # 3 - trzech misjonarzy po lewej stronie
        # 4 - kanibal po lewej stronie
        # 5 - dwóch kanibali po lewej stronie
        # 6 - trzech kanibali po lewej stronie
        # 7 - misjonarz po prawej stronie
        # 8 - dwóch misjonarzy po prawej stronie
        # 9 - trzech misjonarzy po prawej stronie
        # 10 - kanibal po prawej stronie
        # 11 - dwóch kanibal po prawej stronie
        # 12 - trzech kanibal po prawej stronie
        boat = state.find('b')
        river = state.find('-')
        if boat < river:
            new_state[0] = -1
        else:
            new_state[0] = 1
        left_m = state.count('m', 0, river)
        if left_m == 3:
            new_state[3] = 1
        elif left_m == 2:
            new_state[2] = 1
        elif left_m == 1:
            new_state[1] = 1

        left_c = state.count('c', 0, river)
        if left_c == 3:
            new_state[6] = 1
        elif left_c == 2:
            new_state[5] = 1
        elif left_c == 1:
            new_state[4] = 1

        right_m = state.count('m', river)
        if right_m == 3:
            new_state[9] = 1
        elif right_m == 2:
            new_state[8] = 1
        elif right_m == 1:
            new_state[7] = 1

        right_c = state.count('c', river)
        if right_c == 3:
            new_state[12] = 1
        elif right_c == 2:
            new_state[11] = 1
        elif right_c == 1:
            new_state[10] = 1

        return np.reshape(np.array(new_state), -1)

    def reverse_translate_state(state) -> str:
        new_state = []

        if state[0] == -1:
            new_state.append('b')

        if state[6] == 1:
            new_state.append('ccc')
        elif state[5] == 1:
            new_state.append('cc')
        elif state[4] == 1:
            new_state.append('c')

        if state[3] == 1:
            new_state.append('mmm')
        elif state[2] == 1:
            new_state.append('mm')
        elif state[1] == 1:
            new_state.append('m')

        new_state.append('-')
        if state[0] == 1:
            new_state.append('b')

        if state[12] == 1:
            new_state.append('ccc')
        elif state[11] == 1:
            new_state.append('cc')
        elif state[10] == 1:
            new_state.append('c')

        if state[9] == 1:
            new_state.append('mmm')
        elif state[8] == 1:
            new_state.append('mm')
        elif state[7] == 1:
            new_state.append('m')

        return ''.join(map(str, new_state))

    def get_illegal_actions(state):
        state = reverse_translate_state(state)
        legal = list(map(lambda x: translate[x], env.get_possible_actions(state)))
        return [a for a in range(5) if a not in legal]

    def build_model(initial_state, state_size, action_size):
        keras.backend.clear_session()
        model = Sequential()
        model.add(Input(shape=np.reshape(initial_state, (-1)).shape))
        model.add(Dense(state_size ** (action_size / 2), activation='relu'))
        model.add(Dense(state_size ** (action_size / 2), activation='relu'))
        model.add(Dense(state_size ** (action_size / 2), activation='relu'))
        model.add(Dense(state_size ** (action_size / 2), activation='relu'))
        model.add(Dense(action_size))
        model.compile(
            loss=MSE,
            optimizer=Adam(),
        metrics=['accuracy'])
        return model

    agent = None
    initial_state = env.reset()
    translated_initial_state = translate_state(initial_state)

    agent = DDQNAgent(action_size=5, state_size=len(translated_initial_state),
            initial_state=translated_initial_state,
            get_illegal_actions=get_illegal_actions,
            build_model=build_model)

    print("Loading saved models' weights")
    agent.online_model.load_weights("online.tf")
    agent.target_model.load_weights("target.tf")

    if DQNN_LEARN:
        agent.epsilon = 0.9
        agent.epsilon_decay = 0.99
        agent.epsilon_min = 1/14
        agent.learning_rate = 0.001

        done = False
        batch_size = 32
        episodes = 0
        counter = 0

        env.turn_off_display()

        filtered_score_for_plotting = []
        filtered_score = -500
        theoretical_maximum = 500
        episodes = 0
        while filtered_score < theoretical_maximum:
            episodes += 1
            summary = []
            for _ in range(100):
                total_reward = 0
                env_state = env.reset()

                state = translate_state(env_state)

                done = False
                while not done:
                    action = agent.get_action(state)
                    env_action = translate[action]
                    next_state_env, reward, done, score = env.step(env_action)
                    total_reward += reward

                    try:
                        next_state = translate_state(next_state_env)
                    except:
                        # We hit a terminal state, no reason to bother translating
                        print(state, next_state_env)
                        next_state = state

                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                filtered_score = 99/100 * filtered_score + 1/100 * score
                filtered_score_for_plotting.append(filtered_score)

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                
                summary.append(total_reward)
            theoretical_maximum -= 0.1
            print("epoch #{}\texpected reward = {:.3f}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(
                episodes,
                theoretical_maximum,
                np.mean(summary),
                agent.epsilon))    
            agent.update_epsilon_value()

            agent.online_model.save_weights("online.tf", save_format='tf')
            agent.target_model.save_weights("target.tf", save_format='tf')

        plt.plot(filtered_score_for_plotting)
        plt.ylabel("Filtered score")
        plt.xlabel("Epoch")
        plt.show()
        
    env.turn_on_display()
    state = env.reset()
    env_state = translate_state(state)
    agent.turn_off_learning()
    done = False
    while not done:
        action = agent.get_action(env_state)
        env_action = translate[action]
        next_state_env, reward, done, _ = env.step(env_action)
        try:
            next_state = translate_state(next_state_env)
        except:
            # We hit a terminal state, no reason to bother translating
            next_state = state

        state = next_state
        print("Score:", reward)
else:
    print("No algorithm selected")
