import gym
import time
from csv import writer
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from typing import  Union
import numpy as np
import os


class MyMonitor(gym.Wrapper):
    def __init__(self, env: gym.Env, log_dir, name_model) -> None:
        super().__init__(env)
        self.file = open(os.path.join(str(log_dir), name_model+"_metric.csv"), "w+")
        self.log = writer(self.file)
        self.log.writerow(['Episode', 'Timestep', 'Avg Steer', 'Min Reward',
                      'Avg Reward', 'Max Reward', 'Reward Sum', 'Episode Length',
                     'Episode Time','Avg Speed', 'Max Speed', 'Min CTE', 'Avg CTE', 'Max CTE', 'Distance',
                      "Average Throttle", "Max Throttle", "Min Throttle",
                      "Average Absolute CTE", "Min Absolute CTE", "Max Absolute CTE"])

        self.episode = 0
        self.time_step = 0

    def reset(self, **kwargs) -> GymObs:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
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

        if done:
            print("FINISH EPISODE:", self.episode, " time ep: ", round(time.time() - self.start_episode, 4), " ep length tot:",
                  self.episode_len, " avg reward:", round(np.mean(self.rewards), 4), " tot distance:", round(self.distance, 4), "avg throttle:", round(np.mean(self.throttles), 4))

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
        




