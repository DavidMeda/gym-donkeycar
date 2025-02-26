import argparse
from pprint import pprint
import uuid
import gym_donkeycar
import os
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from pyvirtualdisplay import Display
import torch
from torch import nn
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from myCallbacks import SaveOnBestTrainingRewardCallback, StopTrainingOnMaxTimestep, CheckpointCallback
from myWrappers import *


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Torch device avariable:", device)

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

    parser = argparse.ArgumentParser(description="ppo_train")
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument("--test", action="store_true", help="load the trained model and play")
    parser.add_argument("--encoder", action="store_true", help="load the encoder model")
    parser.add_argument("--env_name", type=str, default="donkey-generated-track-v0",
                        help="name of donkey sim environment", choices=env_list)
    parser.add_argument("--server", action="store_true", help="agent run on server, need virtual display")
    parser.add_argument("--host", type=str, default="localhost", help="ip localhost")
    parser.add_argument("--log_dir", type=str, default="./models", help="location of log dir")
    parser.add_argument("--name_model", type=str, default="PPO", help="location of log dir")
    parser.add_argument("--n_step", type=int, default=500, help="port to use for tcp")
    parser.add_argument("--checkpoint", action="store_true", help="Train model from checkpoint")

    args = parser.parse_args()
    display = None
    if args.server:
        display = Display(visible=False, size=(1920, 1080)).start()
        path = args.sim
    else:
        path = "C:\\Users\\david\\Documents\\project\\DonkeySimWin\\v21.07\\donkey_sim.exe"

    env_id = args.env_name
    log_dir = args.log_dir
    name_model = args.name_model
    n_step = args.n_step

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

    # Create the vectorized environment
    env = gym.make(args.env_name, **conf)
    env = MyMonitor(env, log_dir, name_model)
    env = Monitor(env, log_dir)
    # env = ActionClipWrapper(env)
    if args.encoder:
        env = AutoEncoderWrapper(env, os.path.join(log_dir, "encoder_1000.pkl"))
        # env = AutoEncoderHistoryWrapper(env, os.path.join(log_dir, "encoder_1000.pkl"),
        #                                 left_steering=-0.5, right_steering=0.5)
    else:
        env = NormalizeObservation(env)
    # env = SteeringSmoothWrapper(env)

    model = None
    if args.checkpoint:
        model = SAC.load(os.path.join(log_dir, name_model))
        model.set_env(env)
        print("Train from checkpoint at: ", os.path.join(log_dir, name_model))
    else:
        net_arch = {
            "small": [64, 64],
            "medium": [256, 256],
        }

        activation_fn = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU,
                         "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}

        best_param = {"learning_rate":  7.3e-4,
                      "buffer_size": 300000,
                      "batch_size": 256,
                      "ent_coef": 'auto',
                      "gamma": 0.99,
                      "tau": 0.02,
                      # train_freq: 64
                      "train_freq": (1, "episode"),
                    #   gradient_steps: -1
                      "gradient_steps": 64,
                      "learning_starts": 500,
                      "use_sde_at_warmup": True,
                      "use_sde": True,
                      "sde_sample_freq": 64,
                      "policy_kwargs": dict(log_std_init=-2, net_arch=[64, 64])}
        #best_param no encoder
        {'batch_size': 32, 'gamma': 0.9999, 'buffer_size': 5000, 'learning_rate': 0.0001, 'learning_starts': 100, 'tau': 0.0001,
            'gradient_steps': 50, "policy_kwargs": dict(log_std_init=-3, net_arch=net_arch["small"], activation_fn= activation_fn['elu']), 
            'use_sde': True, 'sde_sample_freq': 8}
        #best_param encoder
        {'batch_size': 32, 'gamma': 0.995, 'buffer_size': 300000, 'learning_rate': 5e-05, 'learning_starts': 200, 'tau': 0.02,
            'gradient_steps': 100, "policy_kwargs": dict(log_std_init=-2, net_arch=net_arch["small"]), 
            "use_sde_at_warmup": True, "use_sde": True,"sde_sample_freq": 64, }


        model = SAC("MlpPolicy",env, verbose=0, **best_param)

    auto_save_callback = SaveOnBestTrainingRewardCallback(
        check_freq=10000, log_dir=log_dir, name_model=name_model, verbose=0)
    save_checkpoint = CheckpointCallback(save_freq=100000, save_path=log_dir,
                                            name_prefix=name_model + "_checkpoint", verbose=1)
    callbacks = StopTrainingOnMaxTimestep(n_step, 1)
    # set up model in learning mode with goal number of timesteps to complete
    model.learn(total_timesteps=n_step, callback=[auto_save_callback, save_checkpoint, callbacks])

    # Save the agent
    model.save(os.path.join(log_dir, name_model))
    print("done training")
    if args.server:
        display.stop()

    env.close()
