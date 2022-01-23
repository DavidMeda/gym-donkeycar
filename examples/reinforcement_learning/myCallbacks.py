import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
import os
from typing import Optional

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True


class StopTrainingOnMaxTimestep(BaseCallback):
  """
  Stop the training once a maximum number of timestep are played.

  :param max_timestep: Maximum number of episodes to stop training.
  :param verbose: Select whether to print information about when training ended by reaching ``max_timestep``
  """

  def __init__(self, max_timestep: int, verbose: int = 0):
    super(StopTrainingOnMaxTimestep, self).__init__(verbose=verbose)
    self._total_max_timestep = max_timestep
    self.n_timestep = 0


  def _on_step(self) -> bool:
    done = self.locals.get("done") if self.locals.get("done") is not None else self.locals.get("dones")
    if done:
      self.n_timestep = 0

    self.n_timestep += 1
    continue_training = self.n_timestep < self._total_max_timestep
    if self.verbose > 0 and not continue_training:
        print(
              f" Stopping TRAINING with a total of {self.num_timesteps} steps because the "
              f"{self.locals.get('tb_log_name')} model reached max_timestep={self._total_max_timestep}, "
          )
    return continue_training
