import random
import numpy as np
from collections import deque
from keras.layers import Dense, Conv2D, Flatten, Input, Lambda, add
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras import backend as K


# Note: pass in_keras=False to use this function with raw numbers of numpy arrays for testing
def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error * error / 2
    linear_term = abs(error) - 1 / 2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term


# Double DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, memory_size=6000, epsilon=1.0):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.0005
        self.epsilon = epsilon
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05
        # self.batch_size = 64
        self.batch_size = 512
        self.train_start = self.batch_size * 10
        
        # create replay memory using deque
        self.memory = deque(maxlen=memory_size)
        
        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):

        # CNN Version
        n_filters_1 = 128
        n_filters_2 = 128
        input_grid = Input(shape=(4, 4, 16))

        CNN_10 = Conv2D(n_filters_1, (1, 2), activation='relu', kernel_initializer='he_uniform')(input_grid)
        Flatten_10 = Flatten()(CNN_10)
        CNN_11 = Conv2D(n_filters_2, (1, 2), activation='relu', kernel_initializer='he_uniform')(CNN_10)
        Flatten_11 = Flatten()(CNN_11)
        CNN_12 = Conv2D(n_filters_2, (2, 1), activation='relu', kernel_initializer='he_uniform')(CNN_10)
        Flatten_12 = Flatten()(CNN_12)

        CNN_20 = Conv2D(n_filters_1, (2, 1), activation='relu', kernel_initializer='he_uniform')(input_grid)
        Flatten_20 = Flatten()(CNN_20)
        CNN_21 = Conv2D(n_filters_2, (1, 2), activation='relu', kernel_initializer='he_uniform')(CNN_20)
        Flatten_21 = Flatten()(CNN_21)
        CNN_22 = Conv2D(n_filters_2, (2, 1), activation='relu', kernel_initializer='he_uniform')(CNN_20)
        Flatten_22 = Flatten()(CNN_22)

        CC_1 = concatenate([Flatten_11, Flatten_12, Flatten_21, Flatten_22, Flatten_10, Flatten_20])
        FC_1 = Dense(256, activation='relu', kernel_initializer='he_uniform')(CC_1)

        # network separate state value and advantages
        advantage_fc = Dense(256, activation='relu', kernel_initializer='he_uniform')(FC_1)
        advantage = Dense(self.action_size)(advantage_fc)
        advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                           output_shape=(self.action_size,))(advantage)

        value_fc = Dense(256, activation='relu', kernel_initializer='he_uniform')(FC_1)
        value = Dense(1)(value_fc)
        value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                       output_shape=(self.action_size,))(value)

        # network merged and make Q Value
        q_value = add([value, advantage])
        model = Model(inputs=input_grid, outputs=q_value)
        model.compile(loss=huber_loss, optimizer=Adam(lr=self.learning_rate))

        model.summary()

        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    # Modified Version: return q_value list
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # return random.randrange(self.action_size)
            return random.sample(range(0, self.action_size), self.action_size)
        else:
            q_value = self.model.predict(state)
            # return np.argmax(q_value[0])
            return np.flip(np.argsort(q_value), axis=1)[0]

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        # Non-PER
        self.memory.append((state, action, reward, next_state, done))

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        # Non-PER
        if len(self.memory) < self.train_start:
            return

        np.random.shuffle(self.memory)
        for _ in range(int(self.train_start / self.batch_size)):
            mini_batch = []
            for _ in range(self.batch_size):
                mini_batch.append(self.memory.popleft())

            # CNN Version
            update_input = np.zeros(
                (self.batch_size, self.state_size, self.state_size, self.state_size * self.state_size))
            update_target = np.zeros(
                (self.batch_size, self.state_size, self.state_size, self.state_size * self.state_size))
            # Normal Version
            # update_input = np.zeros((batch_size, self.state_size))
            # update_target = np.zeros((batch_size, self.state_size))
            action, reward, done = [], [], []

            # Non-PER
            for i in range(self.batch_size):
                update_input[i] = mini_batch[i][0]
                action.append(mini_batch[i][1])
                reward.append(mini_batch[i][2])
                update_target[i] = mini_batch[i][3]
                done.append(mini_batch[i][4])

            target = self.model.predict(update_input)
            target_next = self.model.predict(update_target)
            target_val = self.target_model.predict(update_target)

            for i in range(self.batch_size):
                # like Q Learning, get maximum Q value at s'
                # But from target model
                if done[i]:
                    target[i][action[i]] = reward[i]
                else:
                    # the key point of Double DQN
                    # selection of action is from model
                    # update is from target model
                    a = np.argmax(target_next[i])
                    target[i][action[i]] = reward[i] + self.discount_factor * (
                        target_val[i][a])

            # make minibatch which includes target q value and predicted q value
            # and do the model fit!
            self.model.fit(update_input, target, batch_size=self.batch_size,
                           epochs=1, verbose=0)
        self.memory = deque(maxlen=6000)

            # if self.epsilon > self.epsilon_min:
            #     self.epsilon *= self.epsilon_decay

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.target_model.set_weights(self.model.get_weights())

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

