import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random

from collections import defaultdict
from minigrid.envs import crossing
from minigrid.wrappers import SymbolicObsWrapper

SEED = 42

MAX_TRIALS = 5
N_EPISODES = 5_000
MAX_STEPS = 100
NUM_ACTIONS = 3


GAMMA_LIST = [0.975]
ALPHA_LIST = [0.1, 0.2, 0.3]
EPSILON_LIST = [0.01, 0.05, 0.1, 0.2]

GAMMA = None
ALPHA = None
EPSILON = None

# Action Space Representation
# 0 - Left : Turn Left
# 1 - Right : Turn Right
# 2 - Forward: Move Forward 

def encode_state(obs):
    return (tuple(obs['image'].flatten()), obs['direction'])

def get_termination_reason(env):
    pos = env.unwrapped.agent_pos
    cell = env.unwrapped.grid.get(*pos)
    if cell is None:
        return "empty"
    return cell.type


"""
PARAMETERS
    alpha: step size; alpha in (0, 1]
    
INITIALIZE
    Q(s, a) arbitarily
    
FOR EACH episode:
    Initialize state s
    
    REPEAT for each step in episode:
        Choose action a from s using policy derived from Q (eps-greedy)
        Take action a, observe r, s'
        a' = argmax Q(s)
        
        Q(s, a) += alpha * (r + gamma * Q(s', a') - Q(s, a))
        s = s'
        
        IF s is terminal
            break
"""
def qlearn(env, trial_num, trial_seed):
    Q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
    
    rewards_log = []
    steps_log = []
    
    # print(f"\n--- Starting Trial {trial_num + 1}/{MAX_TRIALS} (Seed: {SEED}) ---")

    for episode in range(N_EPISODES): 
        obs, _ = env.reset(seed=trial_seed)
        current_state = encode_state(obs)

        episode_reward = 0 # Reward accumulated during the episode
        episode_steps = 0 # Steps for the current episode

        for _ in range(MAX_STEPS): 
            # Select an action derived from epsilon greedy policy
            # action = np.random.randint(NUM_ACTIONS) if np.random.rand() < EPSILON else np.argmax(Q[current_state])
            action = np.random.randint(NUM_ACTIONS) if np.random.rand() < EPSILON \
                else int(np.random.choice(np.where(Q[current_state] == Q[current_state].max())[0]))

            # Have the agent take the step and gather the outputs
            # Use the next observation to get the next state (s')
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = encode_state(next_obs)

            # argmax_a' Q[s'][a'] = max{ Q[s'] }
            Q[current_state][action] += ALPHA * (reward + GAMMA * np.max(Q[next_state]) - Q[current_state][action])
    
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


if __name__ == "__main__":
    # env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
    # env = gym.make("MiniGrid-LavaCrossingS9N1-v0") # DEBUG
    env_name = "MiniGrid-LavaCrossingS9N2-v0"
    # env_name = "MiniGrid-LavaCrossingS11N5-v0"
    env = gym.make(env_name)
    MAX_STEPS = env.unwrapped.max_steps
    env = SymbolicObsWrapper(env)
    env.reset(seed=SEED)

    # for each parameter combination
    count = 0
    for eps in EPSILON_LIST:
        for gamma in GAMMA_LIST:
            for alpha in ALPHA_LIST:
                # run 50 trails:
                for trial_idx in range(MAX_TRIALS):
                    # to store the trials results:
                    all_trials_rewards = []
                    all_trials_steps = []
                    
                    # set hyperparameters
                    EPSILON = eps
                    GAMMA = gamma
                    ALPHA = alpha

                    # run QL and collect the results:
                    print(f"\n--- Starting Trial {trial_idx + 1}/{MAX_TRIALS} (Seed: {SEED}) (gamma = {gamma}, epsilon = {eps}, alpha = {alpha})---")
                    rewards_log, steps_log = qlearn(env, trial_idx, SEED)

                    all_trials_rewards.append(rewards_log)
                    all_trials_steps.append(steps_log)

                env.close()

                # plot:
                np_all_rewards = np.array(all_trials_rewards)
                np_all_steps = np.array(all_trials_steps)

                # need to find the average and std:
                mean_rewards_per_episode = np.mean(np_all_rewards, axis=0)
                std_rewards_per_episode = np.std(np_all_rewards, axis=0)

                mean_steps_per_episode = np.mean(np_all_steps, axis=0)
                std_steps_per_episode = np.std(np_all_steps, axis=0)

                # requirements said: Take the average reward from episode 1
                episodes_axis = np.arange(1, N_EPISODES + 1)
                plt.figure(figsize=(14, 6))

                # draw average reward:
                plt.subplot(1, 2, 1)
                plt.plot(episodes_axis, mean_rewards_per_episode, label=f'Mean Reward (Trials={MAX_TRIALS})')
                plt.fill_between(episodes_axis, 
                                    mean_rewards_per_episode - std_rewards_per_episode, 
                                    mean_rewards_per_episode + std_rewards_per_episode,
                                    alpha=0.2, label='Std Dev Reward')
                
                plt.xlabel('Episode')
                plt.ylabel('Average Reward')
                plt.title(f'Average Reward per Episode')
                plt.grid(True)
                plt.legend()

                # draw average steps size:
                plt.subplot(1, 2, 2)
                plt.plot(episodes_axis, mean_steps_per_episode, label=f'Mean Steps (Trials={MAX_TRIALS})', color='orange')
                plt.fill_between(episodes_axis, mean_steps_per_episode - std_steps_per_episode,
                                                mean_steps_per_episode + std_steps_per_episode,
                                                color='orange', alpha=0.2, label='Std Dev Steps')
                plt.xlabel('Episode')
                plt.ylabel('Average Steps')
                plt.title(f'Average Steps per Episode')
                plt.grid(True)
                plt.legend()

                plt.suptitle(f"Q-learning: α={ALPHA}, ε={EPSILON}, γ={GAMMA} ({env_name}), Fix Seed", fontsize=14)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                import os
                os.makedirs('images/tune/qlearn', exist_ok=True)
                plt.savefig(f'images/tune/qlearn/qlearn_{MAX_TRIALS}_{N_EPISODES}_count{count + 1}.png', dpi=300, bbox_inches='tight')
                # plt.show()
                print(f"Plot count {count + 1} saved, continuing program.")
                count += 1