import sys
# setting path
sys.path.append('../reinforcement_learning')
import argparse
import uuid
import gym_donkeycar 
import os
import gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from pyvirtualdisplay import Display
import torch
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from myCallbacks import SaveOnBestTrainingRewardCallback, StopTrainingOnMaxTimestep, CheckpointCallback
from myWrappers import MyMonitor, NormalizeObservation, AutoEncoderWrapper



def make_env(env_id, rank, conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id, conf=conf)
        env.seed(seed + rank)
        env.reset()
        return env

    #set_global_seeds(seed)
    return _init


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

    parser = argparse.ArgumentParser(description="TD3_train")
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument("--test", action="store_true", help="load the trained model and play")
    parser.add_argument("--env_name", type=str, default="donkey-generated-track-v0",help="name of donkey sim environment", choices=env_list)
    parser.add_argument("--server", action="store_true", help="agent run on server, need virtual display")
    parser.add_argument("--host", type=str, default="localhost", help="ip localhost")
    parser.add_argument("--log_dir", type=str, default="./models", help="location of log dir")
    parser.add_argument("--name_model", type=str, default="TD3", help="location of log dir")
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
        "racer_name": "TD3",
        "bio": "Learning to drive w TD3 RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 1.5,
        "headless":args.server,
        "log_level": 40
    }

    if args.test:

        # Make an environment test our trained policy
        env = gym.make(args.env_name, **conf)
        env = MyMonitor(env, log_dir, name_model+"_test")
        env = NormalizeObservation(env)
       # env = AutoEncoderWrapper(env, os.path.join(log_dir, "encoder_1000.pkl"))
        env = Monitor(env, log_dir)
        env = DummyVecEnv([lambda: env])

        model = TD3.load(os.path.join(log_dir, name_model))
        print("Loaded model\n", "-" * 30, "\n", model.policy,"\n", "-" * 30)

        mean_reward = 0.0
        for _ in range(5):
            time_step = 0
            obs = env.reset()
            done = False
            rewards = []
            while not done:
                time_step += 1
                action, _states = model.predict(obs)
                # print(action)
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
                if time_step >= args.n_step:
                    print(
                        f" Stopping EVALUATION with a total of {time_step} steps because the TD3 model reached max_timestep={args.n_step}")
                    done = True
            mean_reward += np.sum(rewards)
        mean_reward = mean_reward / 5.0
        print("Mean sum reward ", mean_reward)
        print("DONE TEST")

    else:
        # Create the vectorized environment
        env = gym.make(args.env_name, **conf)
        env = MyMonitor(env, log_dir, name_model)
        env = NormalizeObservation(env)
        #env = AutoEncoderWrapper(env, os.path.join(log_dir, "encoder_1000.pkl"))
        env = Monitor(env, log_dir)
        env = DummyVecEnv([lambda: env])
        
        print("ENV CREATE")
        # Multiprocess n_env in 1 process run asynchronus
        # env = make_vec_env(env_id=args.env_name, n_envs=2, seed=444, monitor_dir=log_dir, env_kwargs=conf)
        model = None
        if args.checkpoint:
            model = TD3.load(os.path.join(log_dir, name_model))
            print("Train from checkpoint at: ", os.path.join(log_dir, name_model))
        else:
            # create cnn policy
            # buffer_size_best = 5000
            best_param = {'batch_size': 32, 'buffer_size': 1000, 'learning_starts': 2500,
                          'gamma': 0.9999, 'tau': 0.001, 'learning_rate': 1e-05, 'gradient_steps': 256}

            model = TD3("MlpPolicy", env, verbose=1,  **best_param)
        
        auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir, name_model=name_model, verbose=0)
        save_checkpoint = CheckpointCallback(save_freq=50000, save_path=log_dir,
                                             name_prefix=name_model + "_checkpoint", verbose=1)
        callbacks = StopTrainingOnMaxTimestep(n_step, 1)
        # set up model in learning mode with goal number of timesteps to complete
        model.learn(total_timesteps=n_step, callback=[auto_save_callback, save_checkpoint, callbacks])

        # Save the agent
        model.save(os.path.join(log_dir,  name_model))
        print("done training")
        if args.server:
            display.stop()

    env.close()
