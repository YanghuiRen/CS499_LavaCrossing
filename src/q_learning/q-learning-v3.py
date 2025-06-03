import numpy
import matplotlib.pyplot as plt
import gymnasium as gym
import random
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minigrid.envs import crossing
from minigrid.wrappers import SymbolicObsWrapper
#from minigrid.wrappers import ImgObsWrapper

MAX_TRIALS = 50
N_EPISODES = 500
MAX_STEPS = 100
NUM_ACTIONS = 3

GAMMA = 0.975
ALPHA = 0.5
EPSILON = 0.5

# Action Space Representation
# 0 - Left : Turn Left
# 1 - Right : Turn Right
# 2 - Forward: Move Forward 

def encode_state(obs):
    
    # Flatten the symbolic image and convert to tuple
    flat_image = obs['image'].flatten()
    return tuple(flat_image.tolist())

def get_termination_reason(env):
    pos = env.unwrapped.agent_pos
    cell = env.unwrapped.grid.get(*pos)
    if cell is None:
        return "empty"
    return cell.type

decay_rate = 0.01

def eps_decay(episode): 
    decay_factor = 1 + decay_rate * episode
    eps = EPSILON / decay_factor
    return eps

def alph_decay(episode):
    alph = max(0.1, ALPHA / (1 + decay_rate * episode))
    return alph

def qlearning(): 
    #Init Q(s,a) = 0
    #Init Q(terminal state, *) = 0

    Q = {} #Stores q-vals
    rewards_log = []
    steps_log = []

    for episode in range(N_EPISODES): 
        obs, _ = env.reset()
        current_state = encode_state(obs)

        episode_reward = 0 # Reward accumulated during the episode
        episode_steps = 0 # Steps for the current episode

        for s in range(MAX_STEPS): 
            #current_state_hash = env.hash(current_state_hash)

            # Generate epsilon value based on decay factor
            epsi = eps_decay(episode)

            # Select an action derived from epsilon greedy policy
            if random.random() < epsi: 
                action = random.randint(0, NUM_ACTIONS - 1)
            else: 
                try: 
                    action = numpy.argmax(Q[current_state])
                except KeyError: 
                    Q[current_state] = numpy.zeros(NUM_ACTIONS)
                    action = numpy.argmax(Q[current_state])

            # Have the agent take the step and gather the outputs
            next_obs, reward, terminated, truncated, res = env.step(action)

            # Use the next observation to get the next state (s')
            next_state = encode_state(next_obs)
            #next_state_hash = env.hash(next_state)

            # Q-Value update
            if current_state not in Q: 
                Q[current_state] = numpy.zeros(NUM_ACTIONS)
            if next_state not in Q: 
                Q[next_state] = numpy.zeros(NUM_ACTIONS)

            # Generate alpha value based on alpha decay
            alpha = alph_decay(episode)

            Q[current_state][action] += ALPHA * (reward + GAMMA * numpy.max(Q[next_state]) - Q[current_state][action])
    
            # Update the reward values for this episode
            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                steps_log.append(episode_steps)
                rewards_log.append(episode_reward)
                break

            # Episode is still going, get ready for the next step
            current_state = next_state

        #Check for termination 
        reason = get_termination_reason(env)
        if reason == "goal":
            print(f"Episode {episode + 1}: Success! Agent reached the goal.")
        elif reason == "lava":
            print(f"Episode {episode + 1}: Failure. Agent fell into lava.")
        elif truncated: print(f"Episode {episode + 1}: Failure. Too many steps.")
        else: print(f"Episode {episode + 1}: Terminated for unknown reasons.")

    #For episode in episodes:
        # Init S
        #For step in episode:
            #Choose A from S using policy derived from Q (episolon greedy)
            #Take action A, observe R and S'
            #Q(S, A) ← Q(S, A) + α[R + γ maxa Q(S′, a) − Q(S, A)]
            #S ← S'
            #Until S is terminal

    return Q, steps_log, rewards_log


if __name__ == "__main__":
    env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
    env = SymbolicObsWrapper(env)
    Q, steps_log, rewards_log = qlearning()
    env.close()

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