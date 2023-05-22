import gym
import numpy as np
from RandomAgent import TimeLimitWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import os

import retro
from utils import *


# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        #env = gym.make(env_id)
        env = retro.make(game=env_id)
        env = TimeLimitWrapper(env, max_steps=2000)
        env = MaxAndSkipEnv(env, 4)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init 

if __name__ == '__main__':
    env_id = "SuperMarioBros-Nes"
    num_cpu = 2  # Number of processes to use
    # Create the vectorized environment
    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]),"tmp/TestMonitor")
    
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log="./board/", 
                learning_rate=0.00003,
                batch_size=4)
    #model = PPO.load("tmp/best_model", env=env)
    print("------------- Start Learning -------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=5000000, callback=callback, tb_log_name="PPO-00003")
    model.save(env_id)
    print("------------- Done Learning -------------")
    env = retro.make(game=env_id)
    env = TimeLimitWrapper(env)
    
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()