"""
file: ppo_train.py
author: Tawn Kramer
date: 13 October 2018
notes: ppo2 test from stable-baselines here:
https://github.com/hill-a/stable-baselines
"""
import argparse
import uuid
import gym_donkeycar
import gym
from stable_baselines3 import PPO
#from stable_baselines3.common import set_global_seeds
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_donkeycar.envs import donkey_env
from pyvirtualdisplay import Display


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
    display = Display(visible=False, size=(1920, 1080))
    display.start()

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
    parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
    parser.add_argument("--env_name", type=str, default="donkey-generated-track-v0", help="name of donkey sim environment", choices=env_list)


    args = parser.parse_args()

    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        exit(1)

    env_id = args.env_name

    conf = {
        "exe_path": "C:\\Users\\david\\Documents\\project\\DonkeySimWin\\donkey_sim.exe",
        #"exe_path":args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "me",
        "font_size": 30,
        "racer_name": "PPO",
        "country": "USA",
        "bio": "Learning to drive w PPO RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 1.5,
    }

    if args.test:

        # Make an environment test our trained policy
        env = gym.make(args.env_name, conf=conf)
        env = DummyVecEnv([lambda: env])

        model = PPO.load("models/ppo_donkey")
        print("Loaded model")
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        # print("Evaluation\n", mean_reward,"\n",std_reward,"\n")
        obs = env.reset()
        for i in range(1000):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

        print("done testing")

    else:

        # make gym env
        #env = gym.make(args.env_name, conf=conf)

        # Create the vectorized environment
        env = DummyVecEnv([make_env(args.env_name, i, conf=conf) for i in range(4)])

        # create cnn policy
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./models/")

        # set up model in learning mode with goal number of timesteps to complete
        model.learn(total_timesteps=100)

        obs = env.reset()

        for i in range(10):

            action, _states = model.predict(obs)

            obs, rewards, dones, info = env.step(action)

            try:
                env.render()
            except Exception as e:
                print(e)
                print("failure in render, continuing...")

            if i % 100 == 0:
                print("saving...")
                model.save("ppo_donkey")

        # Save the agent
        model.save("ppo_donkey")
        print("done training")

    env.close()
    display.stop()