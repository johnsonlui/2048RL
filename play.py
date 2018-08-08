from c2048 import Game, random_play, push
from DDQN_keras_CNNs import DoubleDQNAgent
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import deepcopy

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
agent = DoubleDQNAgent(env.n_features[0], env.n_actions, epsilon=0)
agent.load_weights('myDQN_2048_bestavgscore.h5')

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

        # Save this for next reward checking
        state_tmp = deepcopy(next_state)

        # State Transformation
        next_state = transform_input(next_state)
        next_state = np.array(next_state, dtype=np.float32).reshape(1, 4, 4, 16)

        state = next_state
        steps += 1

        if done:
            # every episode, plot the play time
            score = env.score
            scores.append(score)
            episodes.append(episode)
            if(score > max_score):
                max_score = score
                env.display()
            if(len(scores) > 100 and np.mean(scores[-100:]) > max_avg_score):
                max_avg_score = np.mean(scores[-100:])
            print("episode:", episode, "  score:", score, "  memory length:",
                  len(agent.memory), "  epsilon:", agent.epsilon, "  max score:", max_score, "  max avg score:", max_avg_score)

print("Done")


