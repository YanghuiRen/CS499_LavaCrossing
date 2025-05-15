"""
Header File
"""

import gymnasium as gym

from minigrid.wrappers import SymbolicObsWrapper
from minigrid.wrappers import ImgObsWrapper

MAX_TRIALS = 50
N_EPISODES = 100
MAX_STEPS = 100

GAMMA = 0.975
ALPHA = 0.5
EPSILON = 0.3

"""
Generate an epside
"""
def generate_episode(env, policy=None, max_steps=100):
    obs = env.reset()
    episode = []
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        action = policy(obs) if policy != None else env.action_space.sample()
        next_obs, reward, done, truncate, info = env.step(action)
        episode.append(obs, action, reward, next_obs, done)
        obs = next_obs
        steps += 1
        
    return episode