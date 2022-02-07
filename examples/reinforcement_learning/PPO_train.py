import argparse
import uuid
import gym_donkeycar 
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pyvirtualdisplay import Display
import torch
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from myCallbacks import SaveOnBestTrainingRewardCallback, StopTrainingOnMaxTimestep, CheckpointCallback
from myWrappers import *



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
    parser.add_argument("--env_name", type=str, default="donkey-generated-track-v0",help="name of donkey sim environment", choices=env_list)
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

    # if args.sim == "sim_path" and args.multi:
    #     print("you must supply the sim path with --sim when running multiple environments")
    #     exit(1)

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
        "headless":args.server,
        "log_level": 40
    }

    if args.test:
        print("test")
        # conf["max_cte"] = 10
        # # Make an environment test our trained policy
        # env = gym.make(args.env_name, **conf)
        # env = MyMonitor(env, log_dir, name_model+"_test")
        # if args.encoder:
        #     # env = AutoEncoderWrapper(env, os.path.join(log_dir, "encoder_1000.pkl"))
        #     env = AutoEncoderHistoryWrapper(env, os.path.join(log_dir, "encoder_1000.pkl"),
        #                                     left_steering=-0.5, right_steering=0.5)
        # else:
        #     env = NormalizeObservation(env)
        # env = Monitor(env, log_dir)
        # #env = DummyVecEnv([lambda: env])

        # model = PPO.load(os.path.join(log_dir, name_model))
        # model.set_env(env)
        # #print("Loaded model\n", "-" * 30, "\n", model.policy,"\n", "-" * 30)

        # mean_reward = 0.0
        # n_episode = 3
        # unique_time_laps = []
        # exit_right_lane = 0
        # for _ in range(n_episode):
        #     time_step = 0
        #     obs = env.reset()
        #     done = False
        #     rewards = []
        #     laps_time = []
        #     num_lap = 0
            
        #     while not done:
        #         time_step += 1
        #         action, _states = model.predict(obs)
        #         # print(action)
        #         obs, reward, done, info = env.step(action)
        #         if info["time_last_lap"] > 0:
        #             laps_time.append(info["time_last_lap"])
        #         num_lap = info["num_lap"]
        #         rewards.append(reward)
        #         if abs(info["cte"]) > 1.5:
        #             exit_right_lane += 1
        #             print("EXIT RIGHT LANE, tot", exit_right_lane)
                    
        #         # if time_step >= args.n_step:
        #         #     print(
        #         #         f" Stopping EVALUATION with a total of {time_step} steps because the PPO model reached max_timestep={args.n_step}")
        #         #     done = True
        #         if num_lap >= 3:
        #             print(
        #                 f" Stopping EVALUATION with a total of {time_step} steps because the PPO model reached laps={num_lap}")
        #             unique_time_laps.append(np.unique(laps_time))
        #             print("avg time laps: ", round(np.mean(unique_time_laps), 6))
        #             print("exit right lane: ", exit_right_lane)

        #             done = True
        #     mean_reward += np.sum(rewards)
        # mean_reward = mean_reward / n_episode
        # print("Mean sum reward ", mean_reward)
        # print("Mean time laps: ", round(np.mean(unique_time_laps), 6))
        # print("DONE TEST")

    else:
        # Create the vectorized environment
        env = gym.make(args.env_name, **conf)
        env = MyMonitor(env, log_dir, name_model)
        env = Monitor(env, log_dir)
        if args.encoder:
            # env = AutoEncoderWrapper(env, os.path.join(log_dir, "encoder_1000.pkl"))
            env = AutoEncoderHistoryWrapper(env, os.path.join(log_dir, "encoder_1000.pkl"),
                                            left_sterring=-0.5, right_steering=0.5)
        else:
            env = NormalizeObservation(env)
        env = SteeringSmoothWrapper(env)
        

        model = None
        if args.checkpoint:
            model = PPO.load(os.path.join(log_dir, name_model))
            model.set_env(env)
            print("Train from checkpoint at: ", os.path.join(log_dir, name_model))
        else:
            # create cnn policy
            # best_param_Noencoder = {'batch_size': 8, 'n_steps': 256, 'gamma': 0.995, 'learning_rate': 1.6317237858212062e-05, 'ent_coef': 0.006796716848635915,
            #               'clip_range': 0.3, 'n_epochs': 1, 'gae_lambda': 0.92, 'max_grad_norm': 0.6, 'vf_coef': 0.7925006174080169, 'sde_sample_freq': 16}
            #meglio usare 'n_steps': 256, 'batch_size': 256,
            best_param = {'n_steps': 256, 'batch_size': 256, 'gamma': 0.99, 'learning_rate': 1e-3, 'ent_coef': 8.909295283666419e-06, 'clip_range': 0.2,
                          'n_epochs': 5, 'gae_lambda': 0.99, 'max_grad_norm': 2, 'vf_coef': 0.6215998804092693,  'sde_sample_freq': 8}

            model = PPO("MlpPolicy", env, verbose=0, **best_param)
        
        auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir, name_model=name_model, verbose=0)
        save_checkpoint = CheckpointCallback(save_freq=n_step//2, save_path=log_dir,
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
