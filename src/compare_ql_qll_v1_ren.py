import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random

from collections import defaultdict
from minigrid.envs import crossing
from minigrid.wrappers import SymbolicObsWrapper

N_EPISODES = 1500
# we call MAX_STEPS from env.unwrapped.max_steps directly in main()
# MAX_STEPS = 100

GAMMA = 0.975
ALPHA = 0.2
EPSILON = 0.1
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

def qlearn(env, trial_seed): 
    Q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
    
    rewards_log = []
    steps_log = []

    for episode in range(N_EPISODES): 
        obs, _ = env.reset(seed=trial_seed)
        current_state = encode_state(obs)

        episode_reward = 0 # Reward accumulated during the episode
        episode_steps = 0 # Steps for the current episode

        for _ in range(MAX_STEPS): 
            action = np.random.randint(NUM_ACTIONS) if np.random.rand() < EPSILON else int(np.random.choice(np.where(Q[current_state] == Q[current_state].max())[0]))
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

def qlearn_lambda(env, trial_num, trial_seed):
    Q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
    
    # used for plotting graphs
    steps_log = []
    rewards_log = []

    # debug to track goal rate:
    goal_count = 0

    #DEBUG:
    # print(f"\n--- Starting Trial {trial_num + 1}/{MAX_TRIALS} (Seed: {trial_seed}) ---")
    print(f"\n--- Starting Trial {trial_num + 1}/{MAX_TRIALS} (Seed: {SEED}) ---")
    
    for ep_idx in range(N_EPISODES):
        E = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        # init s, a

        # obs, _ = env.reset(seed=trial_seed)
        obs, _ = env.reset(seed=SEED)

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
                if ep_idx < 5 or (ep_idx + 1) % 100 == 0:
                    print(f"Episode {ep_idx+1}: {termination_reason}, reward={episode_reward:.3f}, steps={episode_steps}")


                steps_log.append(episode_steps)
                rewards_log.append(episode_reward)

                if termination_reason == "goal":
                    goal_count += 1
                break
        
    # DEBUG: print out the results:
    print(f"\n===============================")
    print(f"--- Trial {trial_num + 1} Result: ---")
    print(f"Training {N_EPISODES} episodes.")
    print(f"Total reached: {goal_count} times goal")
    print(f"Success rate: {goal_count / N_EPISODES * 100:.2f}%")
    print(f"===============================")
        
    # return Q
    return rewards_log, steps_log


if __name__ == "__main__":
    MAX_TRIALS = 50
    env_name = "MiniGrid-LavaCrossingS9N1-v0"
    # env_name = "MiniGrid-LavaCrossingS9N2-v0"
    # env_name = "MiniGrid-LavaCrossingS11N5-v0"
    env = gym.make(env_name)
    MAX_STEPS = env.unwrapped.max_steps
    
    ql_rewards_all = []
    ql_steps_all = []
    qll_rewards_all = []
    qll_steps_all = []

    for trial_idx in range(MAX_TRIALS):
        print(f"\nQ-learning Trial {trial_idx+1}/{MAX_TRIALS}")
        ql_rewards, ql_steps = qlearn(env, SEED + trial_idx)
        ql_rewards_all.append(ql_rewards)
        ql_steps_all.append(ql_steps)

        print(f"\nQ(λ) Trial {trial_idx+1}/{MAX_TRIALS}")
        qll_rewards, qll_steps = qlearn_lambda(env, trial_idx, SEED + trial_idx)
        qll_rewards_all.append(qll_rewards)
        qll_steps_all.append(qll_steps)

    env.close()

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

    plt.suptitle(f"Q-learning vs Q-learning(λ): α={ALPHA}, ε={EPSILON}, γ={GAMMA}, λ={LAMBDA}, Trials={MAX_TRIALS} Fix Seed", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('src/images/compare/ql_vs_qll_50_comparison_1500_epi_001epsilon.png', dpi=300, bbox_inches='tight')
    plt.show()