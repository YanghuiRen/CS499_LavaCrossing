import gymnasium as gym
import numpy as np
import minigrid
import matplotlib.pyplot as plt

from collections import defaultdict

from minigrid.wrappers import SymbolicObsWrapper
from minigrid.envs import crossing

MAX_TRIALS = 50
N_EPISODES = 1
MAX_STEPS = 100

GAMMA = 0.975
ALPHA = 0.5
EPSILON = 0.3
LAMBDA = 0.9

# env setup variables
SEED = 42

DIR_MAP = { # DEBUG
    0: 'right',
    1: 'down',
    2: 'left',
    3: 'up'
}
ACT_MAP = { # DEBUG
    0: 'left',
    1: 'right',
    2: 'forward'
}

"""
MiniGrid Action Definitions

0: turn left:       'left'
1: turn right:      'right'
2: move forward:    'forward'
"""
NUM_ACTIONS = 3

# Flatten the symbolic image and convert to tuple
def encode_state(obs):
    flat_image = obs['image'].flatten()
    return tuple(flat_image.tolist())

# for debugging purposes
def get_termination_reason(env):
    pos = env.unwrapped.agent_pos
    print(f"Agent pos at termination: {pos}")
    cell = env.unwrapped.grid.get(*pos)
    if cell is None:
        print("Agent is on an empty cell.")
        return "empty"
    return cell.type

def print_grid(env, ep_num):
    grid = env.unwrapped.grid
    dir = env.unwrapped.agent_dir
    agent_pos = env.unwrapped.agent_pos

    print(f"======= Grid Contents: Ep {ep_num} =======")
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
            cell = grid.get(x, y)
            char = "."
            if cell is not None:
                char = cell.type[0].upper()

            # Overlay agent
            if (x, y) == tuple(agent_pos):
                if char == "L":
                    char = "X"  # Agent on lava marked as X
                else:
                    # char = "A"  # Agent on normal cell
                    dir_dict = {0: '>', 1: 'v', 2: '<', 3: '^'}
                    char = dir_dict[dir]
            row.append(char)
        print(" ".join(row))




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
        obs, _ = env.reset(seed=SEED)
        print_grid(env, ep_idx + 1) # DEBUG
        print(f"Initial Orientation: {DIR_MAP[env.unwrapped.agent_dir]}")
        current_state = encode_state(obs)
        current_action = np.random.randint(NUM_ACTIONS)
        
        # needed for plotting
        episode_reward = 0
        episode_steps = 0
        
        
        # for each step in the episode
        for _ in range(MAX_STEPS):
            # choose a' from s' using policy Q (eps-greedy)
                
            next_action = np.random.randint(NUM_ACTIONS) if np.random.rand() <= EPSILON else np.argmax(Q[current_state])
                    
            # agent take an action
            next_obs, reward, done, truncated, _ = env.step(next_action)
            if next_obs is None: # DEBUG
                print(f"WARNING: next_obs is None!!!!!")
            if reward is None: # DEBUG
                print(f"WARNING: reward is None!!!!!")
            
            # use next observation to get next state
            next_state = encode_state(next_obs)
            if next_state is None: # DEBUG
                print(f"WARNING: next_state is None!!!!!")
            
            # compute a*, delta, INC e(s, a)
            optimal_action = np.argmax(Q[next_state])
            if optimal_action is None: # DEBUG
                print(f"WARNING: optimal_action is None!!!!!")
                
            if Q[next_state][optimal_action] is None: # DEBUG
                print(f"WARNING: Q(s', a*) is None!!!!!")
            if Q[next_state][next_action] is None: # DEBUG
                print(f"WARNING: Q(s', a') is None!!!!!")
                
                
            
            if Q[next_state][optimal_action] == Q[next_state][next_action]: # if a' ties for the max, then a* <-- a'
                optimal_action = next_action
            delta = reward + GAMMA * Q[next_state][optimal_action] - Q[current_state][current_action]
            E[current_state][current_action] += 1
            if E[current_state][current_action] is None: # DEBUG
                print("WARNING: e(s, a) is None")
                
                
            pos = tuple(env.unwrapped.agent_pos) # DEBUG
            print(f"\t Agent took action {next_action}: {ACT_MAP[next_action]} in state {pos}: currently facing {DIR_MAP[env.unwrapped.agent_dir]}")
            print_grid(env, ep_idx + 1)    
                
            # Q-value update
            # note indexes in Q, E should align
            for s in Q:
                for a in range(NUM_ACTIONS):
                    # watch here: 
                    # numpy broadcasting should ensure the indicies match, but may not work as intended
                    if Q[s][a] is None: # DEBUG
                        print("WARNING: Q(s, a) is None!!!!!")
                    if E[s][a] is None: # DEBUG
                        print("WARNING: e(s, a) is None!!!!!")
                    Q[s][a] += ALPHA * delta * E[s][a]
                    E[s][a] *= LAMBDA * GAMMA if optimal_action == next_action else 0
                
            
            # update reward values
            episode_reward += reward
            episode_steps += 1
            
            # prep for next episode step
            current_state = next_state
            current_action = next_action
            
            
            # episode termination check
            if done or truncated:
                steps_log.append(episode_steps)
                rewards_log.append(episode_reward)
                break
            # episode still in progress
        
        # episode has ended here
        reason = get_termination_reason(env)
        print(f"\t Termination condition: {reason}")
        if reason == "goal":
            print(f"SUCCESS: Agent completed episode {ep_idx + 1}")
        elif reason == "lava":
            print(f"FAILURE: Agent fell into lava on episode {ep_idx + 1}")
        elif truncated:
            print(f"TIMEOUT: Agent failed to complete episode {ep_idx + 1}")
        else:
            print(f"UNKOWN: Agent failed for unkown reason on episode {ep_idx + 1}: steps at termination: {episode_steps}")
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
    # env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
    env = gym.make("MiniGrid-LavaCrossingS9N1-v0") # DEBUG
    env = SymbolicObsWrapper(env)
    obs, _ = env.reset(seed=SEED)
    qlearn_lambda(env)
    env.close()