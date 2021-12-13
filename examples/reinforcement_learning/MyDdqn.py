"""
file: ddqn.py
author: Felix Yu
date: 2018-09-12
original: https://github.com/flyyufelix/donkey_rl/blob/master/donkey_rl/src/ddqn.py
"""
import argparse
import os
import random
import signal
import sys
import uuid
from collections import deque
import pickle
import cv2
import gym
import time
import gym_donkeycar
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Conv2D, Conv3D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import *
from pyvirtualdisplay import Display



# EPISODES = 10
# img_rows, img_cols = 80, 80
img_rows, img_cols = 120, 160
# Convert image into Black and white
# img_frames = 4  # We stack 4 frames


class MyLoss(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return (1 + tf.math.exp(tf.abs(y_true))) * tf.abs(y_true - y_pred)


class DQNAgent:
    def __init__(self, state_size, action_space, train=True):
        self.t = 0
        self.max_Q = 0
        self.train = train

        # Get size of state and action
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = action_space

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 1e-4
        if self.train:
            self.epsilon = 1.0
            self.initial_epsilon = 1.0
        else:
            self.epsilon = 1e-6
            self.initial_epsilon = 1e-6
        self.epsilon_min = 0.02
        self.batch_size = 64
        self.train_start = 100
        self.explore = 10000

        # Create replay memory using deque
        self.memory = deque(maxlen=10000)

        # Create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv3D(24, (5, 5, 5), strides=(2, 2, 2), padding="same",
                  input_shape=self.state_size))  # 120*160*3*frames
        model.add(Activation("relu"))
        model.add(Conv3D(32, (5, 5, 5), strides=(2, 2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv3D(64, (5, 5, 5), strides=(2, 2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))

        # 15 categorical bins for Steering angles
        model.add(Dense(15, activation="linear"))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam, metrics=['accuracy'])

        return model

    def rgb2gray(self, rgb):
        """
        take a numpy rgb image return a new single channel image converted to greyscale
        """
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def process_image(self, obs):
        # obs = self.rgb2gray(obs)
        obs = cv2.resize(obs, (img_rows, img_cols))
        return obs

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, s_t):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()[0]
        else:
            # print("Return Max Q Prediction")
            q_value = self.model.predict(s_t)

            # Convert q array to steering value
            return linear_unbin(q_value[0])

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)

        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)  # unpack argument
        state_t = np.concatenate(state_t)

        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        self.max_Q = np.max(targets[0])
        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        for i in range(batch_size):
            if terminal[i]:
                targets[i][action_t[i]] = reward_t[i]
            else:
                a = np.argmax(target_val[i])
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][a])
        result = self.model.train_on_batch(state_t, targets)
        return result

    def load_model(self, name):
        self.model.load_weights(name)

    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


# Utils Functions #
def linear_bin(a):
    """
    Convert a value to a categorical array.

    Parameters
    ----------
    a : int or float
        A value between -1 and 1

    Returns
    -------
    list of int
        A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    """
    Convert a categorical array to value.

    See Also
    --------
    linear_bin
    """
    if not len(arr) == 15:
        raise ValueError("Illegal array length, must be 15")
    b = np.argmax(arr)
    a = b * (2 / 14) - 1
    return a


def run_ddqn(args):
    """
    run a DDQN training session, or test it's result, with the donkey simulator
    """

    t = time.time()
    display = Display(visible=False, size=(1920, 1080))
    display.start()
    EPISODES = args.episode
    img_frames = args.stack_frames
    conf = {
        "exe_path":args.sim,
        "host": args.host,
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "Schumacher",
        "font_size": 30,
        "racer_name": "DDQN",
        "country": "USA",
        "bio": "Learning to drive w DDQN RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
    }

    # Construct gym environment. Starts the simulator if path is given.
    env = gym.make(args.env_name, conf=conf)
    # env = Monitor(env, directory="./models/", force=True,)


    # # not working on windows...
    def signal_handler(signal, frame):
        print("catching ctrl+c")
        env.unwrapped.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)

    # Get size of state and action from environment
    state_size = (img_cols, img_rows, 3, img_frames)
    action_space = env.action_space  # Steering and Throttle

    try:
        agent = DQNAgent(state_size, action_space, train=not args.test)

        throttle = args.throttle  # Set throttle as constant value

        if os.path.exists(args.model):
            print("load the saved model")
            agent.load_model(args.model)
        
        name_model = args.model.replace(".h5", "")
        metrics = []

        for e in range(EPISODES):
            print("Start episode: ", e)
            done = False
            obs = env.reset()

            need_frames = img_frames-1
            # logging
            data_episode = []
            episode_len = 0
            start_episode = time.time()

            x_t = obs
            a = (x_t,)
            for _ in range(img_frames - 1):
                a = a + (x_t,)

            s_t = np.stack(a, axis=3)
            # In Keras, need to reshape
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2], s_t.shape[3])  # 1*80*80*4
            # print("s_t ",s_t.shape)
            while not done:
                if need_frames > 1:
                    steering = agent.get_action(s_t)
                    action = [steering, throttle]
                    next_obs, reward, done, info = env.step(action)

                    x_t1 = agent.process_image(next_obs)

                    x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  
                    s_t1 = np.append(x_t1, s_t[:, :, :, :img_frames - 1], axis=3)  

                    need_frames -= 1
                    continue

                # Get action for the current state and go one step in environment
                steering = agent.get_action(s_t)
                action = [steering, throttle]
                next_obs, reward, done, info = env.step(action)
                
                # x_t1 = agent.process_image(next_obs)
                x_t1 = next_obs
                x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], x_t1.shape[2], 1)
                s_t1 = np.append(x_t1, s_t[:, :, :, :, :3], axis=4)
               
                # Save the sample <s, a, r, s'> to the replay memory
                # q-table
                agent.replay_memory(s_t, np.argmax(linear_bin(steering)), reward, s_t1, done)
                agent.update_epsilon()

                if agent.train:
                    agent.train_replay()
                    #STAT
                    s_t = s_t1
                    agent.t = agent.t + 1
                    episode_len = episode_len + 1
                    data_episode.append({"info:": info, "reward": reward, "action": action,
                                        "Q_MAX ": agent.max_Q, "epsilon: ": agent.epsilon})

                # s_t = s_t1
                # agent.t = agent.t + 1
                # episode_len = episode_len + 1
                    if agent.t % 50 == 0:
                        print("EPISODE", e, "TIMESTEP", agent.t, "/ ACTION", action, "/ REWARD",
                            reward, "/ EPISODE LENGTH", episode_len, "/ Q_MAX ", agent.max_Q,)
                        # print(info)
                if done:
                    # Every episode update the target model to be same with model
                    agent.update_target_model()

                    # Save model for each episode
                    if agent.train:
                        agent.save_model(args.model)
                        metrics.append({"episode": e, "time (min)": (time.time() - start_episode)/60.0, "data": data_episode})
                        # print(metrics[-1])
                        data_episode = []

                    print("FINISH episode:", e, " time (min): ", (time.time() - start_episode)/60.0, "  memory length:", len(agent.memory),
                          "  epsilon:", agent.epsilon, " episode length:", episode_len,)

        print("\nTotal time training (min): ", (time.time() - t) / 60.0)
        with open(name_model+ str('_metrics.plk'), 'wb') as fp:
            pickle.dump(metrics, fp)
        
        env.close()
        display.stop()
    except KeyboardInterrupt:
        print("stopping run...")
    finally:
        env.unwrapped.close()
        with open(name_model+ str('_metrics.plk'), 'wb') as fp:
            pickle.dump(metrics, fp)
        # display.stop()


if __name__ == "__main__":

    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0",
        "donkey-thunderhill-track-v0",
        "donkey-circuit-launch-track-v0",
    ]

    parser = argparse.ArgumentParser(description="ddqn")
    parser.add_argument("--sim",type=str,default="manual",help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.")
    parser.add_argument("--model", type=str, default="rl_driver.h5", help="path to model")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="ip localhost")
    parser.add_argument("--test", action="store_true", help="agent uses learned model to navigate env")
    parser.add_argument("--port", type=int, default=9091, help="port to use for websockets")
    parser.add_argument("--throttle", type=float, default=0.3, help="constant throttle for driving")
    parser.add_argument("--env_name", type=str, default="donkey-generated-track-v0", help="name of donkey sim environment", choices=env_list)
    parser.add_argument("--episode", type=int, default=1, help="number of episode for training")
    parser.add_argument("--stack_frames", type=int, default=4, help="number of frame for stack")
    args = parser.parse_args()
    run_ddqn(args)
