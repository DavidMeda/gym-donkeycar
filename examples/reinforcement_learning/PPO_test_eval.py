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
import pandas as pd

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
    parser.add_argument("--encoder", action="store_true", help="load the encoder model")
    parser.add_argument("--env_name", type=str, default="donkey-generated-track-v0",
                        help="name of donkey sim environment", choices=env_list)
    parser.add_argument("--server", action="store_true", help="agent run on server, need virtual display")
    parser.add_argument("--host", type=str, default="localhost", help="ip localhost")
    parser.add_argument("--log_dir", type=str, default="./models", help="location of log dir")
    parser.add_argument("--name_model", type=str, default="PPO", help="location of log dir")
    parser.add_argument("--episode", type=int, default=3, help="port to use for tcp")

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
        "max_cte": 10,
        "headless": args.server,
        "log_level": 40
    }

    conf["max_cte"] = 10
    
    # Make an environment test our trained policy
    env = gym.make(args.env_name, **conf)
    # env = MyMonitor(env, log_dir, name_model + "_test")
    if args.encoder:
        # env = AutoEncoderWrapper(env, os.path.join(log_dir, "encoder_1000.pkl"))
        env = AutoEncoderHistoryWrapper(env, os.path.join(log_dir, "encoder_1000.pkl"),
                                        left_steering=-0.5, right_steering=0.5)
    else:
        env = NormalizeObservation(env)
    env = Monitor(env, log_dir)

    model = PPO.load(os.path.join(log_dir, name_model))
    model.set_env(env)
    #print("Loaded model\n", "-" * 30, "\n", model.policy,"\n", "-" * 30)

    file = open(os.path.join(str(log_dir), name_model + "_metric.csv"), "w+")
    log = writer(file)
    log.writerow(['Episode', 'Timestep', 'Avg Steer', 'Min Reward',
                'Avg Reward', 'Max Reward', 'Reward Sum', 'Episode Length (timestep)',
                'Episode Time', 'Avg Speed', 'Max Speed', 'Min CTE', 'Avg CTE', 'Max CTE', 'Distance',
                "Average Throttle", "Max Throttle", "Min Throttle",
                "Average Absolute CTE", "Min Absolute CTE", "Max Absolute CTE", "Hit", "Num lap", "Avg time lap", "Best time lap", "Exit right lane", "Percent exit right lane "])
    try:
        n_episode = args.episode
        mean_reward = 0.0
        avg_time_lap = 0.0
        time_step_tot = 0
        for episode in range(n_episode):
            
            obs = env.reset()
            done = False
            rewards = []
            laps_time = []
            unique_time_laps = []
            num_lap = 0
            time_step = 0
            start_episode = time.time()
            episode_len = 0
            steers = []
            throttles = []
            rewards = []
            velocities = []
            ctes = []
            ctes_absolute = []
            distance = 0.0
            distance_time = start_episode
            num_lap = 0
            hit = 0
            laps_time = []
            exit_right_lane = 0

            while not done:
                time_step += 1
                action, _states = model.predict(obs)
                # print(action)
                obs, reward, done, info = env.step(action)
                if info["time_last_lap"] > 0:
                    laps_time.append(info["time_last_lap"])
                num_lap = info["num_lap"]
                rewards.append(reward)
                if abs(info["cte"]) > 1.5:
                    exit_right_lane += 1
                
                steers.append(action[0])
                throttles.append(action[1])
                velocities.append(round(info["speed"], 4))
                rewards.append(round(reward, 4))
                ctes.append(round(info["cte"], 4))
                ctes_absolute.append(round(abs(info["cte"]), 4))
                distance += info["speed"] * (time.time() - distance_time)
                distance_time = time.time()
                episode_len += 1
                time_step_tot += 1
                if info["hit"] !="none":
                    hit += 1

                # if time_step >= args.n_step:
                #     print(
                #         f" Stopping EVALUATION with a total of {time_step} steps because the PPO model reached max_timestep={args.n_step}")
                #     done = True
                if num_lap >= 3:
                    print(
                        f" Stopping EVALUATION with TOT {time_step} steps, the PPO model reached laps={num_lap}")
                    done = True
                
                if done:
                    unique_time_laps.append(np.unique(laps_time))
                    avg_time_lap += np.mean(unique_time_laps)
                    perc_exit = round((float(exit_right_lane) / float(episode_len)) * 100, 4)
                    print("FINISH EPISODE:", episode, f"(timestep: {time_step})" , " sum reward:",
                        round(np.sum(rewards), 4), " avg reward:", round(
                            np.mean(rewards), 4), " tot dist:", round(distance, 4),
                        "avg throt:", round(np.mean(throttles), 4), 
                        "avg time laps: ", round(np.mean(unique_time_laps), 6), "Best time laps: ", round(np.min(unique_time_laps), 6),
                        "exit right lane: ", exit_right_lane, f"({perc_exit}%)")

                    
                    log.writerow([episode, time_step_tot, round(np.mean(steers), 4), round(np.min(rewards), 4),
                                    round(np.mean(rewards), 4), round(np.max(rewards), 4), round(np.sum(rewards), 4),
                                    episode_len, round((time.time() - start_episode), 4),
                                    round(np.mean(velocities), 4), round(np.max(velocities), 4),
                                    round(np.min(ctes), 4), round(np.mean(ctes), 4),
                                    round(np.max(ctes), 4), round(distance, 4), round(
                                        np.mean(throttles), 4), round(np.max(throttles), 4),
                                    round(np.min(throttles), 4), round(np.mean(ctes_absolute), 4),
                                round(np.min(ctes_absolute), 4), round(np.max(ctes_absolute), 4), hit, info["num_lap"], round(np.mean(unique_time_laps), 6), 
                                round(np.min(unique_time_laps), 6), exit_right_lane, perc_exit])

            mean_reward += np.sum(rewards)
        mean_reward = mean_reward / n_episode
        avg_time_lap = avg_time_lap /n_episode
        print("\nMean sum reward ", mean_reward)
        print("Mean time laps (sec): ", round(avg_time_lap, 6))
        
    except KeyboardInterrupt:
        print("stopping run...")
    finally:
        file.flush()
        env.close()
       
    dt = pd.read_csv(os.path.join(str(log_dir), name_model + "_metric.csv"))
    print(
        "\nFINAL RESULT",
        "Avg Steer ",dt["Avg Steer"].mean(),
        "Min Reward ",dt["Min Reward"].min(),
        "Avg Reward", dt["Avg Reward"].mean(),
        "Max Reward", dt["Max Reward"].max(),
        "Reward Sum", dt["Reward Sum"].mean(),
        "Avg Speed", dt["Avg Speed"].mean(),
        "Avg Speed", dt["Max Speed"].max(),
        "Min CTE", dt["Min CTE"].min(),
        "Avg CTE", dt["Avg CTE"].mean(),
        "Max CTE", dt["Max CTE"].max(),
        "Avg Distance", dt["Distance"].mean(),
        "Sum Distance", dt["Distance"].sum(),
        "Min Throttle", dt["Min Throttle"].min(),
        "Average Throttle", dt["Average Throttle"].mean(),
        "Max Throttle", dt["Max Throttle"].max(),
        "Avg time lap", dt["Avg time lap"].mean(),
        "Best time lap", dt["Best time lap"].min(),
        "\n"
    )
    print("DONE TEST")