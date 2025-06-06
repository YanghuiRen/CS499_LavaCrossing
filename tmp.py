import gymnasium as gym
import numpy as np
import minigrid
import matplotlib.pyplot as plt
from collections import defaultdict

from minigrid.wrappers import SymbolicObsWrapper
from minigrid.envs import crossing


MAX_TRIALS = 50
N_EPISODES = 100_000
# we call MAX_STEPS from env.unwrapped.max_steps directly in main()
# MAX_STEPS = 100

GAMMA = 0.975
ALPHA = 0.2
EPSILON = 0.1
LAMBDA = 0.8

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

def encode_state(obs):
    return (tuple(obs['image'].flatten()), obs['direction'])

# Choose a' from s' using policy derived from Q (e.g., ε-greedy)
def select_action(Q, state, epsilon=EPSILON):
    if np.random.rand() < epsilon:
        return np.random.randint(NUM_ACTIONS)
    return int(np.random.choice(np.where(Q[state] == Q[state].max())[0]))

        
# for debugging purposes
def get_termination_reason(env):
    pos = env.unwrapped.agent_pos
    # print(f"Agent pos at termination: {pos}")
    # cell = env.unwrapped.grid.get(*pos)
    # if cell is None:
    #     print("Agent is on an empty cell.")
    #     return "empty"
    # return cell.type

def print_grid(env, ep_num):
    grid = env.unwrapped.grid
    dir = env.unwrapped.agent_dir
    agent_pos = env.unwrapped.agent_pos

    # print(f"======= Grid Contents: Ep {ep_num} =======")
    # for y in range(grid.height):
    #     row = []
    #     for x in range(grid.width):
    #         cell = grid.get(x, y)
    #         char = "."
    #         if cell is not None:
    #             char = cell.type[0].upper()

    #         # Overlay agent
    #         if (x, y) == tuple(agent_pos):
    #             if char == "L":
    #                 char = "X"  # Agent on lava marked as X
    #             else:
    #                 # char = "A"  # Agent on normal cell
    #                 dir_dict = {0: '>', 1: 'v', 2: '<', 3: '^'}
    #                 char = dir_dict[dir]
    #         row.append(char)
    #     print(" ".join(row))
    pass




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
        obs, _ = env.reset(seed=trial_seed)

        # DEBUG: (I set print the results for each 500 episodes, we can change it)
        if ep_idx < 2 or (ep_idx + 1) % 500 == 0:
            print_grid(env, ep_idx + 1)
            # print(f"Initial Orientation: {DIR_MAP[env.unwrapped.agent_dir]}")
        # END DEBUG


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
            
            # if next_obs is None: # DEBUG
            #     print(f"WARNING: next_obs is None!!!!!")
            # if reward is None: # DEBUG
            #     print(f"WARNING: reward is None!!!!!")
            
            # use next observation to get next state
            # if next_state is None: # DEBUG
            #     print(f"WARNING: next_state is None!!!!!")
            
            # compute a*, delta, INC e(s, a)
            optimal_action = np.argmax(Q[next_state])
            # if optimal_action is None: # DEBUG
            #     print(f"WARNING: optimal_action is None!!!!!")
                
            # if Q[next_state][optimal_action] is None: # DEBUG
            #     print(f"WARNING: Q(s', a*) is None!!!!!")
            # if Q[next_state][next_action] is None: # DEBUG
            #     print(f"WARNING: Q(s', a') is None!!!!!")
                
                
            
            if Q[next_state][optimal_action] == Q[next_state][next_action]: # if a' ties for the max, then a* <-- a'
                optimal_action = next_action
            delta = reward + GAMMA * Q[next_state][optimal_action] - Q[current_state][current_action]
            E[current_state][current_action] += 1
            # if E[current_state][current_action] is None: # DEBUG
            #     print("WARNING: e(s, a) is None")
                
                
            # pos = tuple(env.unwrapped.agent_pos) # DEBUG
            # print(f"\t Agent took action {next_action}: {ACT_MAP[next_action]} in state {pos}: currently facing {DIR_MAP[env.unwrapped.agent_dir]}")
            # print_grid(env, ep_idx + 1)    
                
            # Q-value update
            # note indexes in Q, E should align
            # should update E for eligibility trace
            for s in list(E.keys()):
                for a in range(NUM_ACTIONS):
                    # watch here: 
                    # numpy broadcasting should ensure the indicies match, but may not work as intended
                    # if Q[s][a] is None: # DEBUG
                    #     print("WARNING: Q(s, a) is None!!!!!")
                    # if E[s][a] is None: # DEBUG
                    #     print("WARNING: e(s, a) is None!!!!!")
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
                
                # DEBUG: record goal counts:
                # print(f"Episode {ep_idx+1}: {termination_reason}, reward={episode_reward:.3f}, steps={episode_steps}")

                # make print more sparse:
                if ep_idx < 5 or (ep_idx + 1) % 100 == 0:
                    print(f"Episode {ep_idx+1}: {termination_reason}, reward={episode_reward:.3f}, steps={episode_steps}")


                steps_log.append(episode_steps)
                rewards_log.append(episode_reward)
                if termination_reason == "goal":
                    goal_count += 1
                break
            # episode still in progress
        
        # episode has ended here
        # reason = get_termination_reason(env)
        # print(f"\t Termination condition: {reason}")
        # if reason == "goal":
        #     # print(f"SUCCESS: Agent completed episode {ep_idx + 1}")
        #     pass
        # elif reason == "lava":
        #     print(f"FAILURE: Agent fell into lava on episode {ep_idx + 1}")
        # elif truncated:
        #     print(f"TIMEOUT: Agent failed to complete episode {ep_idx + 1}")
        # else:
        #     pass
            # print(f"UNKOWN: Agent failed for unkown reason on episode {ep_idx + 1}: steps at termination: {episode_steps}")
        # end of each step in episode loop
    # end of each episode loop
    
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
    # env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
    # env = gym.make("MiniGrid-LavaCrossingS9N1-v0") # DEBUG
    env_name = "MiniGrid-LavaCrossingS9N3-v0"
    # env_name = "MiniGrid-LavaCrossingS11N5-v0"
    env = gym.make(env_name)
    MAX_STEPS = env.unwrapped.max_steps
    env = SymbolicObsWrapper(env)
    env.reset(seed=SEED)

    # to store the trials results:
    all_trials_rewards = []
    all_trials_steps = []

    # run 50 trails:
    for trial_idx in range(MAX_TRIALS):
        # I think we need this for part B to set different Lava layout
        # trial_seed = SEED + trial_idx
        # rewards_log, steps_log = qlearn_lambda(env, trial_idx, trial_seed)


        # run qll and collect the results:
        rewards_log, steps_log = qlearn_lambda(env, trial_idx, SEED)

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

    plt.suptitle(f"Q-learning(λ): α={ALPHA}, ε={EPSILON}, λ={LAMBDA}, γ={GAMMA} ({env_name}), Fix Seed", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    import os
    os.makedirs('src/images/trials/qll', exist_ok=True)
    plt.savefig(f'src/images/trials/qll/qll_{MAX_TRIALS}_{N_EPISODES}.png', dpi=300, bbox_inches='tight')
    plt.show()
