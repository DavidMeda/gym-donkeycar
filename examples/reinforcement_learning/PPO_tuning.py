import pandas as pd
from torch import nn as nn
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import optuna
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from time import sleep
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from myWrappers import MyMonitor, NormalizeObservation, AutoEncoderWrapper
from stable_baselines3.common.callbacks import EvalCallback
from myCallbacks import StopTrainingOnMaxTimestep
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from pyvirtualdisplay import Display
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import gym
import os
import gym_donkeycar
import uuid
import argparse
import sys
# setting path
sys.path.append('../reinforcement_learning')


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.
    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2])
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    #lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    use_sde = trial.suggest_categorical('use_sde', [False, True])
    sde_sample_freq = -1
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
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "use_sde": use_sde,
        "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }
    print(result)
    # path_ = "/content/MYgdrive/MyDrive/ColabNotebooks/models/"
    env = gym.make(args.env_name, **conf)
    env = MyMonitor(env, args.log_dir , "PPO_tuning_train")
    env = AutoEncoderWrapper(env, os.path.join(args.log_dir, "encoder_1000_large.pkl"))
    #env = NormalizeObservation(env)
    model = PPO("MlpPolicy", env, **result)

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
    env_eval = MyMonitor(env_eval, args.log_dir , "PPO_tuning_test")
    env_eval = AutoEncoderWrapper(env_eval, os.path.join(args.log_dir, "encoder_1000_large.pkl"))
    #env_eval = NormalizeObservation(env_eval)

    mean_reward = 0.0
    for _ in range(3):
        time_step = 0
        obs = env_eval.reset()
        done = False
        rewards = []
        while not done:
            time_step += 1
            action, _states = model.predict(obs)
            obs, reward, done, info = env_eval.step(action)
            rewards.append(reward)
            if time_step >= args.n_step / 2:
                print(
                    f" Stopping EVALUATION with a total of {time_step} steps because the PPO model reached max_timestep={args.n_step/2}")
                done = True
        mean_reward += np.sum(rewards)
    mean_reward = mean_reward / 3.0
    print("Mean reward ", mean_reward)

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
    parser.add_argument("--env_name", type=str, default="donkey-generated-track-v0", help="name of donkey sim environment")
    parser.add_argument("--server", action="store_true", help="agent run on server, need virtual display")
    parser.add_argument("--host", type=str, default="localhost", help="ip localhost")
    parser.add_argument("--log_dir", type=str, default="./models/", help="location of log dir")
    parser.add_argument("--name_model", type=str, default="PPO", help="location of log dir")
    parser.add_argument("--n_step", type=int, default=1000, help="port to use for tcp")

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
        "log_level": 40
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
    study.optimize(sample_ppo_params, n_trials=30)
    trial = study.best_trial
    print("\n\nBEST Reward: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(str(args.log_dir) + str(args.name_model) + "_history.png")

    fig1 = optuna.visualization.plot_param_importances(study)
    fig1.write_image(str(args.log_dir) + str(args.name_model) + "_param_importances.png")
    print("STOP")
    print("done tuning")
    if args.server:
        display.stop()
