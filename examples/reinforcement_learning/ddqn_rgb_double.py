"""
file: ddqn.py
author: Felix Yu
date: 2018-09-12
original: https://github.com/flyyufelix/donkey_rl/blob/master/donkey_rl/src/ddqn.py
"""
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import signal
import sys
import uuid
import time
from collections import deque
import pickle
import cv2
import gym
from tensorflow.python.keras import metrics
from gym.envs.registration import register
import gym_donkeycar
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Conv3D, Dense, Flatten, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import *
from pyvirtualdisplay import Display
from csv import writer


img_rows, img_cols = 120, 160

class DDQNAgent:
    def __init__(self, state_size, action_space, train=True):
        self.t = 0
        self.max_Q = [0,0]
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
        #self.memory = deque(maxlen=10000)
        self.memory = deque(maxlen=20000)

        # Create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.update_target_model()

    def build_model(self):
        img_in = Input(shape=self.state_size, name='img_in')
        x = img_in
        x = Conv3D(24, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu")(x)
        x = Conv3D(32, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu")(x)
        x = Conv3D(64, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu")(x)
        x = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding="same", activation="relu")(x)
        x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding="same", activation="relu")(x)
        x = Activation("relu")(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu', name='dense_1')(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(100, activation='relu', name='dense_2')(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(50, activation='relu', name='dense_3')(x)
        x = Dropout(rate=0.2)(x)

        steering_out = Dense(15, activation='linear', name='steering_out')(x)
        throttle_out = Dense(9, activation='sigmoid', name='throttle_out')(x)
        model = Model(inputs=[img_in], outputs=[steering_out, throttle_out])
        adam = Adam(lr=self.learning_rate)
        model.compile(loss="mse", optimizer=adam)

        return model

    def process_image(self, image):
        img = image.astype(np.float16)
        img /= 255.0
        return img

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, s_t):
        actions = []

        if np.random.rand() <= self.epsilon:
            actions.append(self.action_space.sample()[0])
            actions.append(self.action_space.sample()[1])
        else:
            # print("Return Max Q Prediction")
            q_value = self.model.predict(s_t)
            # print("get_action() prediction: Steering: ", linear_unbin(q_value[0][0], 15), 
            #     "\tTrottle: ", linear_unbin(q_value[1][0], 9))
            
            # Convert q array to steering value and throttle value
            actions.append(linear_unbin(q_value[0][0], 15))
            actions.append(linear_unbin(q_value[1][0], 9))

        return actions

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
        # split minibatch  of replay memory
        state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        # targets is a list of 2
        targets = self.model.predict(state_t)
        self.max_Q = [np.max(targets[0]), np.max(targets[1])] 
        # print("maxQ: ", self.max_Q)

        target_val = self.model.predict(state_t1)
        target_val_ = self.target_model.predict(state_t1)
        for i in range(batch_size):
            if terminal[i]:
                targets[0][i][action_t[i][0]] = reward_t[i]
                targets[1][i][action_t[i][1]] = reward_t[i]
            else:
                a = np.argmax(target_val[0][i])
                b = np.argmax(target_val[1][i])
                
                targets[0][i][action_t[i][0]] = reward_t[i] + (self.discount_factor * (target_val_[0][i][a]))
                targets[1][i][action_t[i][1]] = reward_t[i] + (self.discount_factor * (target_val_[1][i][b]))

        self.model.train_on_batch(state_t, targets)

    def load_model(self, name):
        self.model.load_weights(name)

    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)

# Utils Functions #
def linear_bin(a, n_bin):
    """
    Convert a value to a categorical array.

    Parameters
    ----------
    a : int or float

    Returns
    -------
    list of int
        A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / (n_bin-1)))
    arr = np.zeros(n_bin)
    arr[int(b)] = 1
    return arr

def linear_unbin(arr, n_bin):
    """
    Convert a categorical array to value.

    See Also
    --------
    linear_bin
    """
    if not len(arr) == n_bin:
        raise ValueError("Illegal array length, must be 15")
    b = np.argmax(arr)
    a = b * (2 / (n_bin-1)) - 1
    return a


def run_ddqn(args):
    """
    run a DDQN training session, or test it's result, with the donkey simulator
    """

    # display = Display(visible=False, size=(1920, 1080))
    # display.start()
    EPISODES = args.episode
    img_frames = args.stack_frames
    conf = {
        # "exe_path": "D:\\DonkeySimWin\\DonkeySimWin2\\DonkeySimWin\\donkey_sim.exe",
        "exe_path": "C:\\Users\\david\\Documents\\project\\DonkeySimWin\\donkey_sim.exe",
        #"exe_path":args.sim,
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
        "max_cte": 1.5,
        "log_level": 40
    }

    # Construct gym environment. Starts the simulator if path is given.
    env = gym.make(args.env_name, conf=conf)

    # # not working on windows...
    def signal_handler(signal, frame):
        print("catching ctrl+c")
        env.unwrapped.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)

    # Get size of state and action from environment
    state_size = (img_rows, img_cols,  3, img_frames)
    action_space = env.action_space  # Steering and Throttle
    t = time.time()
    try:
        agent = DDQNAgent(state_size, action_space, train=not args.test)

        if os.path.exists(args.model):
            print("load the saved model")
            agent.load_model(args.model)
        agent.model.summary()
        
        name_model = args.model.replace(".h5", "")
        file = open(str(name_model)+"_metric.csv", "w+")
        log = writer(file)
        log.writerow(['Episode','Timestep', 'Avg Steer', 'Min Reward', 
        'Avg Reward', 'Max Reward','Reward Sum', 'Episode Length',  
        'Max Q steering', 'Max Q throttle', 'Epsilon','Episode Time', 
        'Avg Speed','Max Speed','Min CTE','Avg CTE','Max CTE','Distance', 
        "Average Throttle", "Max Throttle", "Min Throttle",
        "Average Absolute CTE", "Min Absolute CTE", "Max Absolute CTE"])

        for e in range(EPISODES):
            print("START EPISODE: ", e)
            done = False
            obs = env.reset()

            # Stats
            start_episode = time.time()
            episode_len = 0
            steers = []
            throttles = []
            rewards =[]
            velocities = []
            ctes = []
            ctes_absolute = []
            distance = 0.0
            distance_time = start_episode

            need_frames = img_frames-1
            x_t = agent.process_image(obs)
            print("img reset size:",x_t.shape, "min", np.min(x_t), "max", np.max(x_t))

            a = (x_t,)
            for _ in range(img_frames - 1):
                a = a + (x_t,)

            s_t = np.stack(a, axis=3)
            # In Keras, need to reshape
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2], s_t.shape[3])  # 1*80*80*4
            # print("s_t ",s_t.shape)
            while not done:
                if need_frames > 1:
                    action = agent.get_action(s_t)
                    next_obs, reward, done, info = env.step(action)

                    x_t1 = agent.process_image(next_obs)
                    x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], x_t1.shape[2], 1)  
                    s_t1 = np.append(x_t1, s_t[:, :, :, :img_frames - 1], axis=4) 

                    need_frames -= 1
                    continue

                # Get action for the current state and go one step in environment
                action = agent.get_action(s_t)
                next_obs, reward, done, info = env.step(action)
                
                x_t1 = agent.process_image(next_obs)
                print("img size:",x_t1.shape, "min", np.min(x_t1), "max", np.max(x_t1))

                x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], x_t1.shape[2], 1)
                s_t1 = np.append(x_t1, s_t[:, :, :, :, :3], axis=4)
               
                # Save the sample <s, a, r, s'> to the replay memory
                steering_bin = np.argmax(linear_bin(action[0], 15))
                throttle_bin = np.argmax(linear_bin(action[1], 9))
                agent.replay_memory(s_t, [steering_bin, throttle_bin],  reward, s_t1, done)
                agent.update_epsilon()

                if agent.train:
                    agent.train_replay()
                    # stats
                    steers.append(action[0])
                    throttles.append(action[1])
                    velocities.append(round(info["speed"], 4))
                    rewards.append(round(reward,4))
                    ctes.append(round(info["cte"], 4))
                    ctes_absolute.append(round(abs(info["cte"]), 4))
                    distance += info["speed"]*(time.time()-distance_time)
                    distance_time = time.time()

                s_t = s_t1
                agent.t = agent.t + 1
                episode_len = episode_len + 1
                if agent.t % 50 == 0:
                    print(" episode", e, "/ timestep", agent.t, "/ reward",
                        round(reward, 4), "/ episode len", episode_len, "/ q_max [steer, throt] ", agent.max_Q,)

                if done:
                    # Every episode update the target model to be same with model
                    agent.update_target_model()

                    # Save model for each episode
                    if agent.train:
                        agent.save_model(args.model)
                        # stat save
                        log.writerow([e,agent.t, round(np.mean(steers), 4), round(np.min(rewards), 4), 
                            round(np.mean(rewards), 4), round(np.max(rewards), 4), round(np.sum(rewards),4), 
                            episode_len, agent.max_Q[0], agent.max_Q[1], agent.epsilon, round((time.time() - start_episode), 4),
                            round(np.mean(velocities), 4), round(np.max(velocities), 4), round(np.min(ctes), 4), round(np.mean(ctes), 4),
                            round(np.max(ctes), 4), round(distance, 4),  round(np.mean(throttles), 4),round(np.max(throttles), 4),
                            round(np.min(throttles), 4), round(np.mean(ctes_absolute), 4), round(np.min(ctes_absolute), 4), round(np.max(ctes_absolute), 2)    ])

                    print("FINISH EPISODE:", e, " time ep: ", round(time.time() - start_episode,4), " epsilon:", round(agent.epsilon, 4), " ep length tot:", episode_len," avg reward:", round(np.mean(rewards),4)," tot distance:", round(distance, 4), "avg throttle:", round(np.mean(throttles), 4))

        print("\nTotal time training (min): ", (time.time() - t) / 60.0)
        file.flush()
        env.close()
        display.stop()
    except KeyboardInterrupt:
        print("stopping run...")
    finally:
        env.close()
        file.flush()


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
    parser.add_argument("--episode", type=int, default=4, help="number of episode for training")
    parser.add_argument("--stack_frames", type=int, default=4, help="number of frame for stack")
    args = parser.parse_args()
    run_ddqn(args)
