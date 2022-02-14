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
        self.steering_diff = 0.15-1e-5
        self.num_history = num_history
        self.history = np.zeros((1,2 * self.num_history), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([np.float32(self.left_sterring), np.float32(self.min_throttle)]),
            high=np.array([np.float32(self.right_steering), np.float32(self.max_throttle)]),
            dtype=np.float32
            )
        self.observation_space = gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1,self.ae.z_size + 2 * self.num_history),
            dtype=np.float32
            )
    
    def reset(self, **kwargs) -> GymObs:
        obs = self.ae.encode_from_raw_image(self.env.reset()[:, :, ::-1]).flatten()
        obs = np.reshape(obs, (1, self.ae.z_size))
        self.history = np.zeros((1,2 * self.num_history), dtype=np.float32)
        observation = np.concatenate((obs, self.history), axis=-1)
        return observation

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        obs, reward, done, info = self.env.step(action)
        obs = self.ae.encode_from_raw_image(obs[:, :, ::-1]).flatten()
        obs = np.reshape(obs, (1, self.ae.z_size))
        if action[1] == 0.0:
            action[1] = 0.1
        
        # prev_steering = self.history[0, -2]
        # # print("prev_steer", prev_steering)
        # # print("now steer", action[0])
        # diff = np.clip(action[0]- prev_steering, -self.steering_diff, self.steering_diff)
        # # print("diff: ",diff)
        # action[0] = prev_steering + diff
        # # print("NEW steer", action[0], "\n")
        self.history = np.roll(self.history, shift=-2, axis=-1)
        self.history[..., -2:] = action
        observation = np.concatenate((obs, self.history), axis=-1)
        return observation, reward, done, info

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
                      "Average Absolute CTE", "Min Absolute CTE", "Max Absolute CTE"])

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
        # self.num_lap = 0
        # self.laps_time = []
        # self.exit_right_lane = 0

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
        # if abs(info["cte"]) > 1.5:
        #     self.exit_right_lane += 1
        #     print("EXIT RIGHT LANE, tot", self.exit_right_lane)
        # lap_time = info["time_last_lap"]
        # if  lap_time > 0 or self.last_lap_time != self._secondLast_lap_time:
        # self.laps_time.append(info["time_last_lap"])
        #     self.last_lap_time = lap_time
        #     if info["num_lap"] > 1:

        if done:
            # unique_time_laps = np.unique(self.laps_time)
            # print("laps time ", unique_time_laps)
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
                               round(np.min(self.ctes_absolute), 4), round(np.max(self.ctes_absolute), 4)])
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


class SteeringSmoothWrapper(gym.RewardWrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.last_steering = 0.0
        self.velocities = np.ones((1, 10), dtype=np.float32)

    def reset(self, **kwargs):
        self.last_steering = 0.0
        self.velocities = np.ones((1, 10), dtype=np.float32)
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.velocities = np.roll(self.velocities, shift=-1, axis=-1)
        self.velocities[..., -1:] = info["speed"]
        return observation, self.reward(reward, action), done, info

    def reward(self, reward, action):
        angle = action[0]
        import math
        angle_diff = abs(angle - self.last_steering)

        if reward > 0:
            avg_vel = np.mean(self.velocities)
            if avg_vel < 1:
                avg_vel = 1
            if angle_diff > 0.15:
                reward_return = 1 + reward *avg_vel* (1.0 / angle_diff)
                # print(f"CUT reward: {reward_return} - angle= {angle_diff} now {angle} prec {self.last_steering}")
                self.last_steering = angle
                return reward_return
                # reward = reward * 0.1
            a = 1 + (reward * avg_vel**2)
            # print(f"PRO reward: {a} - angle= {angle_diff} avg vel= {avg_vel}")
            self.last_steering = angle
            return a
            # else:
            #     reward = reward * 10
            # print(f"PRO x2 reward: {reward} - angle={angle_diff} now{angle} prec{self.last_steering}")
        # print("PENALIZE reward ", reward)
        self.last_steering = angle
        return reward