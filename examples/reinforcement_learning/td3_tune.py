import argparse
import uuid
import gym_donkeycar
import os
import gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from pyvirtualdisplay import Display
import torch
from stable_baselines3.common.evaluation import evaluate_policy
from myCallbacks import StopTrainingOnMaxTimestep
from stable_baselines3.common.callbacks import EvalCallback
from myWrappers import MyMonitor, NormalizeObservation, AutoEncoderWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from time import sleep

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import optuna
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn
import pandas as pd


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
    #n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [1e5, 2e5, 5e5, 1e6, 2e6 ])
    learning_starts = trial.suggest_loguniform("learning_starts",1e2, 1e4)
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    tau = trial.suggest_categorical("tau", [1e-2, 2e-2, 1e-3, 2e-3, 5e-3, 1e-4])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    gradient_steps = trial.suggest_categorical("gradient_steps", [-1, 10, 50, 100, 256])
    
    result = {
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "learning_starts": learning_starts,
        "tau": tau,
        "gradient_steps": gradient_steps
        
        }
    print(result)
    # path_ = "/content/MYgdrive/MyDrive/ColabNotebooks/models/"
    env = gym.make(args.env_name, **conf)
    env = MyMonitor(env, args.log_dir , "PPO_tuning")
    #env = AutoEncoderWrapper(env, os.path.join(args.log_dir, "encoder_1000.pkl"))
    env = NormalizeObservation(env)
    model = TD3("MlpPolicy", env, **result)

    stopTrainCallback = StopTrainingOnMaxTimestep(args.n_step, 1)
    try:
        model.learn(args.n_step, callback=stopTrainCallback)
        model.env.close()
        
    except (AssertionError, ValueError) as e:
        model.env.close()
    sleep(5)
    print("EVALUATION...")
    
    env_eval = gym.make(args.env_name, **conf)
    env_eval = Monitor(env_eval, args.log_dir)
    env_eval = NormalizeObservation(env_eval)
    #env_eval = AutoEncoderWrapper(env_eval, os.path.join(args.log_dir, "encoder_1000.pkl"))

    mean_reward = 0.0
    for _ in range(5):
        time_step = 0
        obs = env_eval.reset()
        done = False
        rewards = []
        while not done:
            time_step += 1
            action, _states = model.predict(obs)
            obs, reward, done, info = env_eval.step(action)
            rewards.append(reward)
            if time_step >= args.n_step:
                print(
                    f" Stopping EVALUATION with a total of {time_step} steps because the PPO model reached max_timestep={args.n_step}")
                done = True
        mean_reward += np.sum(rewards)
    mean_reward = mean_reward / 5.0
    print("Mean reward ", mean_reward, "\n")

    env_eval.close()

    return mean_reward



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device avariable:", device)

    # Initialize the donkey environment
    # where env_name one of:

    parser = argparse.ArgumentParser(description="ppo_train")
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument("--env_name", type=str, default="donkey-generated-track-v0",help="name of donkey sim environment")
    parser.add_argument("--server", action="store_true", help="agent run on server, need virtual display")
    parser.add_argument("--host", type=str, default="localhost", help="ip localhost")
    parser.add_argument("--log_dir", type=str, default="./models/", help="location of log dir")
    parser.add_argument("--name_model", type=str, default="PPO", help="location of log dir")
    parser.add_argument("--n_step", type=int, default=500, help="port to use for tcp")

    global args
    args = parser.parse_args()
    display = None
    if args.server:
        display = Display(visible=False, size=(1920, 1080)).start()
        path = args.sim
    else:
        path = "C:\\Users\\david\\Documents\\project\\DonkeySimWin\\donkey_sim.exe"

    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        exit(1)

    env_id = args.env_name
    log_dir = args.log_dir
    name_model = args.name_model

    global conf
    conf = {
        "exe_path": path,
        "host": args.host,
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "Schumacher",
        "font_size": 30,
        "racer_name": "PPO",
        "bio": "Learning to drive w PPO RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 1.5,
        "headless": args.server,
        "log_level": 60
    }

    # make gym env
    #env = gym.make(args.env_name, conf=conf)

    # Create the vectorized environment
    #env = gym.make(args.env_name, **conf)
    #env = MyMonitor(env, log_dir, name_model)
    #env = DummyVecEnv([lambda: env])
    #env = make_vec_env(env_id=args.env_name, n_envs=2, seed=444, monitor_dir=log_dir, env_kwargs=conf)
    # env = DummyVecEnv(envs)

    study = optuna.create_study(direction='maximize')
    study.optimize(sample_ppo_params, n_trials=50)
    trial = study.best_trial
    print("\n\nBEST Reward: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    print("STOP")
    print("done tuning")
    if args.server:
        display.stop()

