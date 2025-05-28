import gymnasium as gym
import numpy as np
import minigrid
import matplotlib.pyplot as plt

from collections import defaultdict

from minigrid.wrappers import SymbolicObsWrapper
from minigrid.envs import crossing
# from minigrid.core.constants import OBJECT_TO_IDX

MAX_TRIALS = 50
N_EPISODES = 100
MAX_STEPS = 100

GAMMA = 0.975
ALPHA = 0.5
EPSILON = 0.3
LAMBDA = 0.9

NUM_ACTIONS = 3


def encode_state(obs):
    
    # Flatten the symbolic image and convert to tuple
    flat_image = obs['image'].flatten()
    return tuple(flat_image.tolist())

"""
Q-learning(lambda). Assume the passed in environment is a lava crossing env with a SymbolicObsWrapper.

Initialize Q(s, a) arbitrarily and e(s, a) = 0 for all s,a
Repeat for each episode:
    Initialize s, a
    Repeat for each step in episode:
        Take an action a, observe r, s'
        Choose a' from s' using policy Q derived from eps-greedy
        a* <-- argmax_b Q(s', b) if a' ties for max, a* <-- a'
        delta <-- r + gamma * Q(s', a*) - Q(s, a)
        e(s, a) <-- e(s, a) + 1
        For all s,a:
            Q(s, a) <-- Q(s, a) + alpha * delta * e(s, a)
            If a' = a* then e(s, a) <-- gamma * lambda * e(s, a)
            else e(s, a) = 0
    until s is terminal

==============================
===== Definitions ============
==============================
current_state := s
current_action := a
next_state := s'
next_action := a'

"""
def qlearn_lambda(env):
    Q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
    E = defaultdict(lambda: np.zeros(NUM_ACTIONS))
    
    # used for plotting graphs
    steps_log = []
    rewards_log = []
    
    for ep_idx in range(N_EPISODES):
        # init s, a
        obs, _ = env.reset()
        current_state = encode_state(obs)
        current_action = np.random.randint(NUM_ACTIONS)
        
        # needed for plotting
        episode_reward = 0
        episode_steps = 0
        
        
        # for each step in the episode
        for _ in range(MAX_STEPS):
            # choose a' from s' using policy Q (eps-greedy)
            next_action = None
            
            if np.random.rand() < EPSILON:
                next_action =  np.random.randint(NUM_ACTIONS)
            else:
                next_action = np.argmax(Q[current_state])
                    
            # agent take an action
            next_obs, reward, done, truncated, _ = env.step(next_action)
            
            # use next observation to get next state
            next_state = encode_state(next_obs)
            
            # compute a*, delta, INC e(s, a)
            optimal_action = np.argmax(Q[next_state])
            if Q[next_state][optimal_action] == Q[next_state][next_action]: # if a' ties for the max, then a* <-- a'
                optimal_action = next_action
            delta = reward + GAMMA * Q[next_state][optimal_action] - Q[current_state][current_action]
            E[current_state][current_action] += 1
                
            # Q-value update
            # note indexes in Q, E should align
            for s in Q:
                for a in range(NUM_ACTIONS):
                    # watch here: 
                    # numpy broadcasting should ensure the indicies match, but may not work as intended
                    Q[s][a] += ALPHA * delta * E[s][a]
                    E[s][a] *= LAMBDA * GAMMA if optimal_action == next_action else 0
                
            
            # update reward values and log data for plotting
            episode_reward += reward
            episode_steps += 1
            
            if done:
                steps_log.append(episode_steps)
                rewards_log.append(episode_reward)
                print(f"Agent successfully completed episode {ep_idx + 1}.")
                break
            if truncated:
                steps_log.append(episode_steps)
                rewards_log.append(0)
                print(f"Agent did failed to complete episode {ep_idx + 1}.")
                break
            
            # prep for next episode step
            current_state = next_state
            current_action = next_action
            
            # episode still in progress
        # end of each step in episode loop
    # end of each episode loop
        
        
        
        
        
        
    # Plotting Rewards
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards_log, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per Episode')
    plt.grid(True)
    plt.legend()

    # Plotting Steps
    plt.subplot(1, 2, 2)
    plt.plot(steps_log, label='Steps per Episode', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
        
    return Q









if __name__ == "__main__":
    env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
    env = SymbolicObsWrapper(env)
    obs, _ = env.reset(seed=42)
    qlearn_lambda(env)
    env.close()