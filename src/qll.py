import gymnasium as gym
import numpy as np
import minigrid
import matplotlib.pyplot as plt

from collections import defaultdict

from minigrid.wrappers import SymbolicObsWrapper
from minigrid.envs import crossing
from minigrid.core.constants import OBJECT_TO_IDX

MAX_TRIALS = 50
N_EPISODES = 1000
MAX_STEPS = 300

GAMMA = 0.975
ALPHA = 0.3
LAMBDA = 0.95

NUM_ACTIONS = 3

EPSILON_START, EPSILON_END, DECAY = 1.0, 0.05, 0.995
EPSILON = EPSILON_START

# used for plotting graphs
steps_log = []
rewards_log = []


def encode_state(obs):
    """
    encode_state(obs):
        return tuple, hashtable for Q-table;
    """
    image = obs['image'] # should return shape (S, S, 3)
    agent_idx = OBJECT_TO_IDX['agent']
    goal_idx = OBJECT_TO_IDX['goal']
    lava_idx = OBJECT_TO_IDX['lava']

    # find agent pos:
    agent_pos = tuple(int(x) for x in np.argwhere(image[:,:,2] == agent_idx)[0])

    # agent direction:
    agent_dir = int(obs['direction']) # 0, 1, 2, 3

    # goal position:
    goal_pos = tuple(int(x) for x in np.argwhere(image[:,:,2] == goal_idx)[0])

    # lava distribution:
    lava_pos_list = [tuple(int(x) for x in pos) for pos in np.argwhere(image[:,:,2] == lava_idx)]
    lava_pos_tuple = tuple(sorted(lava_pos_list))

    return (agent_pos, agent_dir, goal_pos, lava_pos_tuple)

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
current_state_hash := s
current_action := a
next_state := s'
next_action := a'

"""
def qlearn_lambda(env):
    # Q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
    Q = defaultdict(lambda: np.zeros(NUM_ACTIONS, dtype=np.float64))
    eps = EPSILON
    
    for ep_idx in range(N_EPISODES):
        # E = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        E = defaultdict(lambda: np.zeros(NUM_ACTIONS, dtype=np.float64))
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
            # current_state_hash = env.hash(current_state)
            current_state_hash = current_state
            next_action = None
            
            if np.random.rand() < eps:
                next_action =  np.random.randint(NUM_ACTIONS)
            else:
                try:
                    next_action = np.argmax(Q[current_state_hash])
                except:
                    Q[current_state_hash] = np.zeros(NUM_ACTIONS)
                    next_action = np.argmax(Q[current_state_hash])
                    
            # agent take an action
            next_obs, reward, done, truncated, _ = env.step(next_action)
            
            # use next observation to get next state
            # next_state = encode_state(next_obs)
            next_state = current_state if done or truncated else encode_state(next_obs)
            # next_state_hash = env.hash(next_state)
            next_state_hash = next_state
            
            # compute a*, delta, INC e(s, a)
            optimal_action = np.argmax(Q[next_state_hash])
            if Q[next_state_hash][optimal_action] == Q[next_state_hash][next_action]: # if a' ties for the max, then a* <-- a'
                optimal_action = next_action
            # delta = reward + GAMMA * Q[next_state_hash][optimal_action] - Q[current_state][current_action]
            delta = reward + GAMMA * Q[next_state_hash][optimal_action] - Q[current_state_hash][current_action]
            E[current_state_hash][current_action] += 1
                
            # Q-value update
            # note indexes in Q, E should align
            # for idx in Q:
            for idx in list(E.keys()): # stil can update while State appears in E but not in Q
                # watch here: 
                # numpy broadcasting should ensure the indicies match, but may not work as intended
                Q[idx] += ALPHA * delta * E[idx]
                # E[idx] *= LAMBDA * GAMMA if optimal_action == next_action else 0
                if next_action == optimal_action:
                    E[idx] *= GAMMA * LAMBDA
                else:
                    E[idx] = np.zeros(NUM_ACTIONS)
      
            
            # update reward values and log data for plotting
            episode_reward += reward
            episode_steps += 1
            
            if done:
                # steps_log[ep_idx] = episode_steps
                # rewards_log[ep_idx] = episode_reward
                steps_log.append(episode_steps)
                rewards_log.append(episode_reward)
                # print("Agent successfully completed an episode.")
                break
            if truncated:
                # steps_log[ep_idx] = episode_steps
                steps_log.append(episode_steps)
                # rewards_log = 0
                rewards_log.append(0)
                # print("Agent did not successfully complete an episode. ")
                break

            # prep for next episode step
            current_state = next_state
            current_action = next_action

        eps = max(EPSILON_END, eps * DECAY)
            
            # episode still in progress
        # end of each step in episode loop
        
    return Q




def main():
    env_id = "MiniGrid-LavaCrossingS9N1-v0"
    env = gym.make(env_id, render_mode=None)
    env = SymbolicObsWrapper(env)
    obs, _ = env.reset(seed=42)

    Q = qlearn_lambda(env)
    total_states = len(Q)
    max_q        = max(np.max(v) for v in Q.values())
    avg_reward   = sum(rewards_log[-10:]) / 10
    avg_steps    = sum(steps_log[-10:]) / 10
    succ_last50 = sum(r > 0 for r in rewards_log[-50:]) / 50

    print("\n=== Training Hy-pra ===")
    print(f"Env         : {env_id}")
    print(f"Episodes    : {N_EPISODES}")
    print(f"Max steps   : {MAX_STEPS}")
    print(f"Gamma       : {GAMMA}")
    print(f"Alpha       : {ALPHA}")
    print(f"Epsilon start, end, decay: {EPSILON_START, EPSILON_END, DECAY}")
    print(f"Lambda      : {LAMBDA}")

    print("\n Results:")
    print(f"Unique states learned : {total_states}")
    print(f"Max Q value           : {max_q:.3f}")
    print(f"Avg reward (last 10)  : {avg_reward:.3f}")
    print(f"Avg steps  (last 10)  : {avg_steps:.1f}")
    print(f"Success rate (last 50): {succ_last50:.1%}")



if __name__ == "__main__":
    main()