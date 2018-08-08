from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from RL2048.DDQN_keras_CNNs import DoubleDQNAgent
import matplotlib.pyplot as plt
import numpy as np
import math
from copy import deepcopy
import datetime
print(datetime.datetime.now())
#
final_score = []
max_score = 0
max_avg_score = 0
steps = 0
EPISODES = 200001
arrow = [Keys.ARROW_LEFT, Keys.ARROW_UP, Keys.ARROW_RIGHT, Keys.ARROW_DOWN]


# agent = DoubleDQNAgent(env.n_features[0], env.n_actions, 6000)
agent = DoubleDQNAgent(4, 4, epsilon=0)
agent.load_weights('myDQN_2048_bestavgscore.h5')

scores, episodes = [], []

for episode in range(EPISODES):
    score = 0

    driver = webdriver.Chrome(executable_path=r"C:\\C_Desktop\\DataScience\\source\\bin\\chromedriver.exe")
    driver.get("https://gabrielecirulli.github.io/2048/")
    time.sleep(0.5)
    driver.find_element_by_class_name('notice-close-button').click()
    state = get_grid(driver)

    # State Transformation
    state_tmp = deepcopy(state)
    state = transform_input(state)
    state = np.array(state, dtype=np.float32).reshape(1, 4, 4, 16)

    body = driver.find_element_by_tag_name('body')

    done = False
    while not done:
        # if agent.render:
        # if True:
        #    env.render()

        # get action for the current state and go one step in environment
        action_list = agent.get_action(state)
        i = 0
        for action in action_list:
            body.send_keys(arrow[action])
            time.sleep(0.05)
            while True:
                try:
                    next_state = get_grid(driver)
                    break
                except:
                    continue
            if done:
                break
            if np.array_equal(state_tmp, next_state):
                if i < 3:
                    i += 1
                else:
                    done = True
                    break
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
            score = int(driver.find_element_by_class_name('score-container').text)
            scores.append(score)
            episodes.append(episode)
            if score > max_score:
                max_score = score
            if len(scores) > 100 and np.mean(scores[-100:]) > max_avg_score:
                max_avg_score = np.mean(scores[-100:])
            print(datetime.datetime.now(), "episode:", episode, "  score:", score, "  max score:", max_score, "  max avg score:", max_avg_score)
            if 2048 in state_tmp:
                print("2048!!")
                time.sleep(10)
            time.sleep(1)
            driver.close()
print("Done")

#driver.close()
