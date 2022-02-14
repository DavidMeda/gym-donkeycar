import pandas as pd
from torch import nn as nn
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import optuna
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from time import sleep
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from myWrappers import *
from stable_baselines3.common.callbacks import EvalCallback
from myCallbacks import StopTrainingOnMaxTimestep
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from pyvirtualdisplay import Display
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
import gym
import os
import gym_donkeycar
import uuid
import argparse
from pprint import pprint
import sys
# setting path
sys.path.append('../reinforcement_learning')


def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.
    :param trial:
    :return:
    """

    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    buffer_size = trial.suggest_categorical("buffer_size", [100000, 200000, 300000, 500000])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2])
    learning_starts = trial.suggest_categorical("learning_starts", [100, 200, 300, 400, 500, 1000])
    lr_schedule = "constant"
    tau = trial.suggest_categorical("tau", [1e-2, 2e-2, 1e-3, 2e-3, 5e-3, 1e-4])
    gradient_steps = trial.suggest_categorical("gradient_steps", [-1, 10, 50, 100, 256])

    # Uncomment to enable learning rate schedule
    #lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    log_std_init = trial.suggest_categorical("log_std_init", [-4, -3, -2, -1, 1])
    use_sde = trial.suggest_categorical('use_sde', [False, True])
    sde_sample_freq = -1
    use_sde_at_warmup = False
    # Uncomment for gSDE (continuous action)
    # Orthogonal initialization
    ortho_init = False
    ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    # activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    # if batch_size > n_steps:
    #     batch_size = n_steps
    if use_sde:
        sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [8, 16, 32, 64, 128])
        use_sde_at_warmup = True
    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    activation_fn = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU,
                     "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    result = {
        "batch_size": batch_size,
        "gamma": gamma,
        "buffer_size": buffer_size,
        "ent_coef": 'auto',
        "gamma":gamma,
        "tau": tau,
        "learning_starts" : learning_starts,
        "gradient_steps": gradient_steps,
        "train_freq": [1, "episode"],
        "learning_rate": learning_rate,
        "use_sde_at_warmup": use_sde_at_warmup,
        "use_sde": use_sde,
        "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            )
        }


    pprint(result)
    # path_ = "/content/MYgdrive/MyDrive/ColabNotebooks/models/"
    env = gym.make(args.env_name, **conf)
    env = MyMonitor(env, args.log_dir , "SAC_tuning")
    # env = AutoEncoderHistoryWrapper(env, os.path.join(log_dir, "encoder_1000.pkl"),
    #                                 left_steering=-0.5, right_steering=0.5)
    env = AutoEncoderWrapper(env, os.path.join(log_dir, "encoder_1000.pkl"))
    env = SteeringSmoothWrapper(env)
    #env = NormalizeObservation(env)
    model = SAC("MlpPolicy", env=env, verbose=0, **result)

    stopTrainCallback = StopTrainingOnMaxTimestep(args.n_step, 1)
    try:
        model.learn(args.n_step, callback=stopTrainCallback)
        model.env.close()

    except (AssertionError, ValueError) as e:
        model.env.close()
    sleep(5)
    print("EVALUATION...")

    env_eval = gym.make(args.env_name, **conf)
    #env_eval = Monitor(env_eval, args.log_dir)
    # env_eval = AutoEncoderHistoryWrapper(env_eval, os.path.join(
    #     args.log_dir, "encoder_1000.pkl"), left_steering=-0.5, right_steering=0.5)
    env_eval = AutoEncoderWrapper(env_eval, os.path.join(log_dir, "encoder_1000.pkl"))
    env_eval = SteeringSmoothWrapper(env_eval)

    #env_eval = NormalizeObservation(env_eval)
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
                    f" Stopping EVALUATION with a total of {time_step} steps because the SAC model reached max_timestep={args.n_step}")
                done = True
        mean_reward += np.sum(rewards)
    mean_reward = mean_reward / 5.0
    print("Mean reward ", mean_reward)

    env_eval.close()

    return mean_reward


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device avariable:", device)

    # Initialize the donkey environment
    # where env_name one of:

    parser = argparse.ArgumentParser(description="SAC_train")
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument("--env_name", type=str, default="donkey-generated-track-v0", help="name of donkey sim environment")
    parser.add_argument("--server", action="store_true", help="agent run on server, need virtual display")
    parser.add_argument("--host", type=str, default="localhost", help="ip localhost")
    parser.add_argument("--log_dir", type=str, default="./models/", help="location of log dir")
    parser.add_argument("--name_model", type=str, default="SAC", help="location of log dir")
    parser.add_argument("--n_step", type=int, default=500, help="port to use for tcp")

    global args
    args = parser.parse_args()
    display = None
    if args.server:
        display = Display(visible=False, size=(1920, 1080)).start()
        path = args.sim
    else:
        path = "C:\\Users\\david\\Documents\\project\\DonkeySimWin\\donkey_sim.exe"

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
        "racer_name": "SAC",
        "bio": "Learning to drive w SAC RL",
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
    study.optimize(sample_sac_params, n_trials=50)
    trial = study.best_trial
    print("\n\nBEST Reward: {}".format(trial.value))
    print("Best hyperparameters: \n", (trial.params))
    pprint(trial.params)
    print("STOP")
    print("done tuning")
    if args.server:
        display.stop()
