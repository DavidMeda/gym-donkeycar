import matplotlib.pyplot as plt
import pygame
import gym
import time
from csv import writer
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from typing import  Union
import numpy as np
import os
from typing import Optional
from autoencoder import load_ae
import gym.wrappers


class AutoEncoderWrapper(gym.Wrapper):
    def __init__(self, env, ae_path):
        super().__init__(env)
        self.env = env
        assert ae_path is not None
        self.ae = load_ae(ae_path)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.ae.z_size,), dtype=np.float32)

    def reset(self):
        return self.ae.encode_from_raw_image(self.env.reset()[:, :, ::-1]).flatten()

    def step(self, action):
        obs, reward, done, infos = self.env.step(action)
        return self.ae.encode_from_raw_image(obs[:, :, ::-1]).flatten(), reward, done, infos

class AutoEncoderHistoryWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, ae_path, num_history=10, max_throttle=1.0, min_throtthle=0.0, left_steering=-1.0, right_steering=1.0) -> None:
        super().__init__(env)
        self.env = env
        self.ae = load_ae(ae_path)
        self.max_throttle = max_throttle
        self.min_throttle = min_throtthle 
        self.left_sterring = left_steering
        self.right_steering = right_steering 
        self.num_command = 2
        self.steering_diff = 0.15-1e-5
        self.num_history = num_history
        self.history = np.zeros((1, self.num_command*self.num_history), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([np.float32(self.left_sterring), np.float32(self.min_throttle)]),
            high=np.array([np.float32(self.right_steering), np.float32(self.max_throttle)]),
            dtype=np.float32
            )
        self.observation_space = gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1,self.ae.z_size + self.num_command*self.num_history),
            dtype=np.float32
            )
    
    def reset(self, **kwargs) -> GymObs:
        obs = self.ae.encode_from_raw_image(self.env.reset()[:, :, ::-1]).flatten()
        obs = np.reshape(obs, (1, self.ae.z_size))
        self.history = np.zeros((1, self.num_command*self.num_history), dtype=np.float32)
        observation = np.concatenate((obs, self.history), axis=-1)
        return observation

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        # last_steering = self.history[0, -2]
        # diff = np.clip(action[0] - last_steering, -self.steering_diff, self.steering_diff)
        # #print(f"pred {action[0]} - last {last_steering} - now {last_steering + diff}")
        # action[0] = last_steering + diff
        self.history = np.roll(self.history, shift=-self.num_command, axis=-1)
        self.history[..., -self.num_command:] = action

        obs, reward, done, info = self.env.step(action)
        obs = self.ae.encode_from_raw_image(obs[:, :, ::-1]).flatten()
        obs = np.reshape(obs, (1, self.ae.z_size))
        # if action[1] == 0.0:
        #     action[1] = 0.1
        #prev_steering = self.history[0, -1]
        # # print("prev_steer", prev_steering)
        # # print("now steer", action[0])
        #diff = np.clip(action[0]- prev_steering, -self.steering_diff, self.steering_diff)
        # # print("diff: ",diff)
        #action[0] = prev_steering + diff
        # # print("NEW steer", action[0], "\n")
        
        observation = np.concatenate((obs, self.history), axis=-1)
        return observation, self.new_reward(reward), done, info

    def new_reward(self, reward):
        
        steering = self.history[0, 1*-self.num_command]
        last_steering = self.history[0, 2*-self.num_command]
        # print(f"Original reward: {reward} steer {steering} last_steer {last_steering} ")
        angle_diff = abs(steering - last_steering)
        penalization = 0
        if angle_diff > 0.15:
            penalization = 2*angle_diff
        if reward < 0:
            for i in range(len(self.history)):
                if self.history[0, i * -self.num_command] >= 0.001:
                    reward = -20 * self.history[0, i * -self.num_command]
                    break
        if reward > 0 and penalization > 0:
            reward = 0
            # print(f" penalize: {penalization} ang_dif {angle_diff}")
        reward -= penalization
        return reward

from threading import Event, Thread

class DonkeyViewWrapper(gym.ObservationWrapper):
    def __init__(self, env, path_ae):
        gym.ObservationWrapper.__init__(self, env)
        self.env = env
        self.ae = load_ae(path_ae)
        self.display_width = 640
        self.display_height = 320
        self.game_display = None
        self.raw_observation = None
        self.decoded_surface = None
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.YELLOW = (255, 255, 0)
        self.BLUE = (0, 0, 255)
        self.reconstructed_image = None
        self.ae_observation = None
        self.game_over = False
        self.start_process()

    def main_loop(self):
        pygame.init()
        self.game_display = pygame.display.set_mode((self.display_width, self.display_height))
        pygame.display.set_caption('Agent View')
        clock = pygame.time.Clock()

        self.game_display.fill(self.WHITE)
        while not self.game_over:
            self.upateScreen()
            clock.tick(30)

    def start_process(self):
        """Start main loop process."""
        self.process = Thread(target=self.main_loop)
        self.process.daemon = True
        self.process.start()

    def pilImageToSurface(self, pilImage):
        pilImage = pilImage.resize((640, 320))
        return pygame.image.fromstring(
            pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

    def upateScreen(self):
        self.game_display.fill((0, 0, 0))
        if self.reconstructed_image is not None:
            pygame_surface = self.pilImageToSurface(self.reconstructed_image)
            self.game_display.blit(pygame_surface, pygame_surface.get_rect(center=(320, 160)))
            pygame.display.update()

    def observation(self, observation):
        # logger.info(observation.shape)
        vae_dim = self.ae.z_size
        self.ae_observation = observation.copy()[0, :vae_dim]
        encoded = self.ae_observation.reshape(1, self.ae.z_size)
        # encoded = self.ae_observation[:, :vae_dim]
        self.reconstructed_image = self.ae.decode(encoded)
        self.reconstructed_image = np.reshape(self.reconstructed_image, (80,160,3))
        plt.imshow(self.reconstructed_image)
        plt.show()
        return observation


class SteeringSmoothWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.last_steering = 0.0
        self.speed = 0.0
        self.throttle = 0.0
        # self.velocities = np.ones((1, 10), dtype=np.float32)

    def reset(self, **kwargs):
        self.last_steering = 0.0
        self.hrottle = 0.0
        self.speed = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.speed = info["speed"]
        return observation, self.reward(reward, action), done, info

    def reward(self, reward, action):
        angle = action[0]
        self.throttle = action[1]
        angle_diff = abs(angle - self.last_steering)
        self.last_steering = angle
        contrib_throttle = 0
        if reward > 0:
            if self.throttle >=0.3 and self.throttle<=0.6:
                contrib_throttle = self.throttle*4
            elif self.throttle < 0.1: 
                contrib_throttle = -1

            if angle_diff > 0.15:
                reward_return =  reward + contrib_throttle + self.speed - (3*angle_diff)
                # print(f"PEN reward: {reward_return} - rew {reward} cont_thr {contrib_throttle} th{self.throttle}")
                return reward_return
            
            reward_return = (reward + contrib_throttle + self.speed) 
            # print(f"PRO reward: {reward_return} - rew {reward} cont_thr {contrib_throttle} th{self.throttle} speed {self.speed}")
            return reward_return
        return reward

class ActionClipWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.last_steering = 0.0

    def action(self, action):
        diff = np.clip(action[0] - self.last_steering, -0.15, 0.15)
        #print(f"pred {action[0]} - last {last_steering} - now {last_steering + diff}")
        action[0] = self.last_steering + diff
        #steering = (.4*action[0])+(.6*self.last_steering)
        # print(f"CLIP ACTION: new {steering} old {action[0]} prec {self.last_steering}")
        #action[0] = steering
        self.last_steering = action[0]
        return action

class MyMonitor(gym.Wrapper):
    def __init__(self, env: gym.Env, log_dir, name_model) -> None:
        super().__init__(env)
        selfenv = env
        self.file = open(os.path.join(str(log_dir), name_model+"_metric.csv"), "w+")
        self.log = writer(self.file)
        self.log.writerow(['Episode', 'Timestep', 'Avg Steer', 'Min Reward',
                      'Avg Reward', 'Max Reward', 'Reward Sum', 'Episode Length (timestep)',
                     'Episode Time','Avg Speed', 'Max Speed', 'Min CTE', 'Avg CTE', 'Max CTE', 'Distance',
                      "Average Throttle", "Max Throttle", "Min Throttle",
                           "Average Absolute CTE", "Min Absolute CTE", "Max Absolute CTE", "Hit", "Num lap", "Avg time lap", "Best time lap", ])

        self.episode = 0
        self.time_step = 0

    def reset(self, **kwargs) -> GymObs:
        self.start_episode = time.time()
        self.episode_len = 0
        self.steers = []
        self.throttles = []
        self.rewards = []
        self.velocities = []
        self.ctes = []
        self.ctes_absolute = []
        self.distance = 0.0
        self.distance_time = self.start_episode
        self.episode +=1
        self.num_lap = 0
        self.laps_time = []
        self.hit = 0

        return self.env.reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        observation, reward, done, info = self.env.step(action)
        self.time_step += 1
        self.steers.append(action[0])
        self.throttles.append(action[1])
        self.velocities.append(round(info["speed"], 4))
        self.rewards.append(round(reward, 4))
        self.ctes.append(round(info["cte"], 4))
        self.ctes_absolute.append(round(abs(info["cte"]), 4))
        self.distance += info["speed"] * (time.time() - self.distance_time)
        self.distance_time = time.time()
        self.episode_len += 1
        self.num_lap = info["num_lap"]
        if info["time_last_lap"] > 0:
            self.laps_time.append(info["time_last_lap"])
        
        if info["hit"] != "none":
            self.hit += 1

        if done:
            unique_time_laps = []
            if self.num_lap > 0:
                unique_time_laps.append(np.unique(self.laps_time))
            else:
                unique_time_laps.append(-1)
            print("FINISH EPISODE:", self.episode, f"(timestep: {self.time_step})" , " timestep ep: ", round(self.episode_len, 4), " sum reward:",
                  round(np.sum(self.rewards), 4), " avg reward:", round(np.mean(self.rewards), 4), " tot dist:", round(self.distance, 4), 
                  "avg throt:", round(np.mean(self.throttles), 4), "num laps: ", info["num_lap"])

            self.log.writerow([self.episode, self.time_step, round(np.mean(self.steers), 4), round(np.min(self.rewards), 4),
                               round(np.mean(self.rewards), 4), round(np.max(self.rewards), 4), round(np.sum(self.rewards), 4),
                               self.episode_len, round((time.time() - self.start_episode), 4),
                               round(np.mean(self.velocities), 4), round(np.max(self.velocities), 4), 
                               round(np.min(self.ctes), 4), round(np.mean(self.ctes), 4),
                               round(np.max(self.ctes), 4), round(self.distance, 4), round(
                                   np.mean(self.throttles), 4), round(np.max(self.throttles), 4),
                               round(np.min(self.throttles), 4), round(np.mean(self.ctes_absolute), 4), 
                               round(np.min(self.ctes_absolute), 4), round(np.max(self.ctes_absolute), 4)],
                               self.hit, round(np.mean(unique_time_laps), 6),round(np.min(unique_time_laps), 6),)
        return observation, reward, done, info
        
    def close(self) -> None:
        """
        Closes the environment
        """
        super(MyMonitor, self).close()
        self.file.flush()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps
        """
        return self.time_step

class NormalizeObservation(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env = env

        
    def reset(self, **kwargs) -> GymObs:
        obs = self.env.reset(**kwargs)
        obs = obs.astype(np.float16)
        obs /= 255.0
        return obs

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        observation, reward, done, info = self.env.step(action)
        observation = observation.astype(np.float16)
        observation /= 255.0
        return observation, reward, done, info

