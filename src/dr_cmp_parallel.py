import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random
import multiprocessing as mp

from functools import partial
from collections import defaultdict
from minigrid.envs import crossing
from minigrid.wrappers import SymbolicObsWrapper

N_EPISODES = 2_000

GAMMA = 0.975
ALPHA = 0.1
EPSILON = 0.05
LAMBDA = 0.8

# env setup variables
SEED = 42

NUM_ACTIONS = 3

def encode_state(obs):
    return (tuple(obs['image'].flatten()), obs['direction'])

def select_action(Q, state, epsilon=EPSILON):
    if np.random.rand() < epsilon:
        return np.random.randint(NUM_ACTIONS)
    return int(np.random.choice(np.where(Q[state] == Q[state].max())[0]))

def run_qlearn_trial(trial_idx, env_name, seed):
    env = gym.make(env_name)
    env = SymbolicObsWrapper(env)
    rewards, steps = qlearn(env, seed)
    env.close()
    return rewards, steps

def run_qll_trial(trial_idx, env_name, seed):
    env = gym.make(env_name)
    env = SymbolicObsWrapper(env)
    rewards, steps = qlearn_lambda(env, seed)
    env.close()
    return rewards, steps

def qlearn(env, trial_seed=None): 
    Q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
    MAX_STEPS = env.unwrapped.max_steps
    
    rewards_log = []
    steps_log = []

    for episode in range(N_EPISODES): 
        obs, _ = env.reset() if trial_seed is None else env.reset(seed=trial_seed)
        current_state = encode_state(obs)

        episode_reward = 0 # Reward accumulated during the episode
        episode_steps = 0 # Steps for the current episode

        for _ in range(MAX_STEPS): 
            action = np.random.randint(NUM_ACTIONS) if np.random.rand() < EPSILON \
                else int(np.random.choice(np.where(Q[current_state] == Q[current_state].max())[0]))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = encode_state(next_obs)
            next_action = np.argmax(Q[next_state])

            # argmax_a' Q[s'][a'] = max{ Q[s'] }
            Q[current_state][action] += ALPHA * (reward + GAMMA * Q[next_state][next_action] - Q[current_state][action])
    
            # Update the reward values for this episode
            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                steps_log.append(episode_steps)
                rewards_log.append(episode_reward)
                break

            # Episode is still going, get ready for the next step
            current_state = next_state

        # Episode has terminated
        if episode % 100 == 0:
            print(f"Finished episode {episode}.")

    return rewards_log, steps_log

def qlearn_lambda(env, trial_seed=None):
    Q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
    MAX_STEPS = env.unwrapped.max_steps
    
    # used for plotting graphs
    steps_log = []
    rewards_log = []

    # debug to track goal rate:
    goal_count = 0
    
    for ep_idx in range(N_EPISODES):
        E = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        # init s, a

        obs, _ = env.reset() if trial_seed is None else env.reset(seed=trial_seed)

        current_state = encode_state(obs)
        current_action = select_action(Q, current_state)
        
        # needed for plotting
        episode_reward = 0
        episode_steps = 0
        
        
        # for each step in the episode
        for _ in range(MAX_STEPS):
            # choose a' from s' using policy Q (eps-greedy)
            # agent take an action
            next_obs, reward, done, truncated, _ = env.step(current_action)
            next_state = encode_state(next_obs) # find next state
            next_action = select_action(Q, next_state)
            
            # compute a*, delta, INC e(s, a)
            optimal_action = np.argmax(Q[next_state])
            
            if Q[next_state][optimal_action] == Q[next_state][next_action]: # if a' ties for the max, then a* <-- a'
                optimal_action = next_action
            delta = reward + GAMMA * Q[next_state][optimal_action] - Q[current_state][current_action]
            E[current_state][current_action] += 1
                
            for s in list(E.keys()):
                for a in range(NUM_ACTIONS):
                    # watch here: 
                    # numpy broadcasting should ensure the indicies match, but may not work as intended
                    Q[s][a] += ALPHA * delta * E[s][a]
                    # we only update the 
                    if next_action == optimal_action:
                        E[s][a] *= GAMMA * LAMBDA
                    else:
                        E[s][a] = 0
                
            
            # update reward values
            episode_reward += reward
            episode_steps += 1
            
            # prep for next episode step
            current_state = next_state
            current_action = next_action
            
            
            # episode termination check
            if done or truncated:
                if truncated:
                    termination_reason = "timeout"
                elif reward > 0:
                    termination_reason = "goal"
                else:
                    termination_reason = "lava"
                
                # make print more sparse:
                if (ep_idx + 1) % 100 == 0:
                    print(f"Episode {ep_idx+1}: {termination_reason}, reward={episode_reward:.3f}, steps={episode_steps}")


                steps_log.append(episode_steps)
                rewards_log.append(episode_reward)

                if termination_reason == "goal":
                    goal_count += 1
                break
        
    return rewards_log, steps_log


if __name__ == "__main__":
    MAX_TRIALS = 50
    env_name = "MiniGrid-LavaCrossingS9N1-v0"
    # env_name = "MiniGrid-LavaCrossingS9N2-v0"
    # env_name = "MiniGrid-LavaCrossingS11N5-v0"
    # env = gym.make(env_name)
    # MAX_STEPS = env.unwrapped.max_steps
        
    # Parallel execution across CPU cores
    ql_results = []
    qll_results = []
    with mp.Pool(mp.cpu_count()) as pool:
        # print("\nRunning Q-learning Trials in Parallel...")
        # ql_results = pool.map(partial(run_qlearn_trial, env_name=env_name, seed=SEED), range(MAX_TRIALS))
        # print("\nRunning Q-learning(λ) Trials in Parallel...")
        # qll_results = pool.map(partial(run_qll_trial, env_name=env_name, seed=SEED), range(MAX_TRIALS))
        args = [(i, env_name, SEED) for i in range(MAX_TRIALS)]
        print("\nRunning Q-learning Trials in Parallel...")
        ql_results = pool.starmap(run_qlearn_trial, args)
        
        print("\nRunning Q-learning(λ) Trials in Parallel...")
        qll_results = pool.starmap(run_qll_trial, args)

    # Unpack results
    ql_rewards_all, ql_steps_all = zip(*ql_results)
    qll_rewards_all, qll_steps_all = zip(*qll_results)
    print(f"ql results - {np.array(ql_results).shape}, qll results - {np.array(qll_results).shape}")
    ql_results = np.array(ql_results)
    qll_results = np.array(qll_results)
    print(f"unzip ql results - {np.array(ql_rewards_all).shape}, unzip qll results - {np.array(ql_steps_all).shape}")
    print(f"unzip ql results - {np.array(qll_rewards_all).shape}, unzip qll results - {np.array(qll_steps_all).shape}")

    ql_rewards_np = np.array(ql_rewards_all)
    ql_steps_np = np.array(ql_steps_all)
    qll_rewards_np = np.array(qll_rewards_all)
    qll_steps_np = np.array(qll_steps_all)

    ql_mean_rewards = np.mean(ql_rewards_np, axis=0)
    ql_std_rewards = np.std(ql_rewards_np, axis=0)
    qll_mean_rewards = np.mean(qll_rewards_np, axis=0)
    qll_std_rewards = np.std(qll_rewards_np, axis=0)

    ql_mean_steps = np.mean(ql_steps_np, axis=0)
    ql_std_steps = np.std(ql_steps_np, axis=0)
    qll_mean_steps = np.mean(qll_steps_np, axis=0)
    qll_std_steps = np.std(qll_steps_np, axis=0)

    episodes_axis = np.arange(1, N_EPISODES + 1)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(episodes_axis, ql_mean_rewards, label='Q-learning', color='#1f77b4')
    plt.fill_between(episodes_axis, 
                    ql_mean_rewards - ql_std_rewards, 
                    ql_mean_rewards + ql_std_rewards, 
                    color='#1f77b4', alpha=0.2, label='Q-learning Std Dev')

    plt.plot(episodes_axis, qll_mean_rewards, label='Q-learning(λ)', color='#ff7f0e')
    plt.fill_between(episodes_axis, 
                    qll_mean_rewards - qll_std_rewards, 
                    qll_mean_rewards + qll_std_rewards, 
                    color='#ff7f0e', alpha=0.2, label='Q-learning(λ) Std Dev')

    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(episodes_axis, ql_mean_steps, label='Q-learning', color='#1f77b4')
    plt.fill_between(episodes_axis, 
                    ql_mean_steps - ql_std_steps, 
                    ql_mean_steps + ql_std_steps, 
                    color='#1f77b4', alpha=0.2, label='Q-learning Std Dev')

    plt.plot(episodes_axis, qll_mean_steps, label='Q-learning(λ)', color='#ff7f0e')
    plt.fill_between(episodes_axis, 
                    qll_mean_steps - qll_std_steps, 
                    qll_mean_steps + qll_std_steps, 
                    color='#ff7f0e', alpha=0.2, label='Q-learning(λ) Std Dev')

    plt.xlabel('Episode')
    plt.ylabel('Average Steps')
    plt.title('Average Steps per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle(f"Q-learning vs Q-learning(λ): α={ALPHA}, ε={EPSILON}, γ={GAMMA}, λ={LAMBDA}, Trials={MAX_TRIALS} Domain Randomization", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    import os
    os.makedirs(f'images/compare/', exist_ok=True)
    plt.savefig('images/compare/dr_eps{EPSILON}_alpha{ALPHA}.png', dpi=300, bbox_inches='tight')
    plt.show()