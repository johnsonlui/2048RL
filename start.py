from c2048 import Game, random_play, push
from DDQN_keras_CNNs import DoubleDQNAgent
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import deepcopy

# convert the input game matrix into corresponding power of 2 matrix.
def transform_input(X):
    power_mat = np.zeros(shape=(1, 4, 4, 16), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            if X[i][j] == 0:
                power_mat[0][i][j][0] = 1.0
            else:
                power = int(math.log(X[i][j], 2))
                power_mat[0][i][j][power] = 1.0
    return power_mat


# find the number of empty cells in the game matrix.
def findEmptyCell(mat):
    count = 0
    for i in range(len(mat)):
        for j in range(len(mat)):
            if mat[i][j] == 0:
                count += 1
    return count


# 2048 game
env = Game()

# Training
final_score = []
max_score = 0
max_avg_score = 0
steps = 0
EPISODES = 200001
update_target_step = 10

# agent = DoubleDQNAgent(env.n_features[0], env.n_actions, 6000)
agent = DoubleDQNAgent(env.n_features[0], env.n_actions)

# try:
#     agent.load_weights('myDQN_2048.h5')
#     print("Loaded weight file")
# except:
#     pass

scores, episodes = [], []

for episode in range(EPISODES):
    score = 0
    state = env.reset()

    # state = np.log(state + 1)
    # state = np.reshape([state], (4, 4, 1))
    # state = np.array([state])

    # State Transformation
    state_tmp = deepcopy(state)
    state = transform_input(state)
    state = np.array(state, dtype=np.float32).reshape(1, 4, 4, 16)

    done = False
    while not done:
        # if agent.render:
        # if True:
        #    env.render()

        # get action for the current state and go one step in environment
        action_list = agent.get_action(state)
        i = 0
        for action in action_list:
            next_state, reward, done, info = env.step(action)
            if done:
                break
            if np.array_equal(state_tmp, next_state):
                if i < 3:
                    i += 1
                else:
                    print("ERROR!")
                continue
            else:
                break

        # Reward Adjustment
        # 1. Empty Cell Reward
        if not done:
            empty_prev = findEmptyCell(state_tmp)
            empty_next = findEmptyCell(next_state)
            reward_empty = empty_next + 1 - empty_prev  # See the difference of empty cell
        else:
            reward_empty = 0

            # 2. Next Max Cell Reward
        prev_max = np.max(state_tmp)
        next_max = np.max(next_state)
        if prev_max != next_max:
            reward_nextMax = math.log(next_max, 2) * 0.1
        else:
            reward_nextMax = 0
        reward_final = reward_empty + reward_nextMax

        # Save this for next reward checking
        state_tmp = deepcopy(next_state)

        # State Transformation
        next_state = transform_input(next_state)
        next_state = np.array(next_state, dtype=np.float32).reshape(1, 4, 4, 16)

        # save the sample <s, a, r, s'> to the replay memory
        agent.append_sample(state, action, reward_final, next_state, done)

        state = next_state
        steps += 1

        # Update epsilon
        if episode > 10000 or (agent.epsilon > 0.1 and steps % 2500 == 0):
            agent.epsilon = agent.epsilon * agent.epsilon_decay

        if done:
            # every episode do the training
            agent.train_model()

            if episode % 20 == 0:
                # every 10 episode update the target model to be same with model
                agent.update_target_model()

            # every episode, plot the play time
            score = env.score
            scores.append(score)
            episodes.append(episode)
            if(score > max_score):
                max_score = score
                agent.save_weights('myDQN_2048_bestscore.h5', overwrite=True)
            if(len(scores) > 100 and np.mean(scores[-100:]) > max_avg_score):
                max_avg_score = np.mean(scores[-100:])
                agent.save_weights('myDQN_2048_bestavgscore.h5', overwrite=True)
            print("episode:", episode, "  score:", score, "  memory length:",
                  len(agent.memory), "  epsilon:", agent.epsilon, "  max score:", max_score, "  max avg score:", max_avg_score)
            if episode % 100 == 0:
                env.display()

print("Done")


agent.save_weights('myDQN_2048.h5', overwrite=True)
