import numpy
import matplotlib.pyplot as plt
import gymnasium as gym
import random
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encode_state import encode_state

from minigrid.envs import crossing
from minigrid.wrappers import SymbolicObsWrapper
#from minigrid.wrappers import ImgObsWrapper

MAX_TRIALS = 50
N_EPISODES = 100
MAX_STEPS = 100
NUM_ACTIONS = 3

GAMMA = 0.975
ALPHA = 0.5
EPSILON = 0.3

# Action Space Representation
# 0 - Left : Turn Left
# 1 - Right : Turn Right
# 2 - Forward: Move Forward 

Q = {} #Stores q-vals
rewards_log = []
steps_log = []

def qlearning(): 
    #Init Q(s,a) = 0
    #Init Q(terminal state, *) = 0

    for episode in range(N_EPISODES): 
        obs, _ = env.reset()
        current_state = encode_state(obs)

        episode_reward = 0 # Reward accumulated during the episode
        episode_steps = 0 # Steps for the current episode

        for s in range(MAX_STEPS): 
            current_state_hash = env.hash(current_state)

            # Select an action derived from epsilon greedy policy
            if random.random() < EPSILON: 
                action = random.randint(0, NUM_ACTIONS - 1)
            else: 
                try: 
                    action = numpy.argmax(Q[current_state_hash])
                except KeyError: 
                    Q[current_state_hash] = numpy.zeros(NUM_ACTIONS)
                    action = numpy.argmax(Q[current_state_hash])

            # Have the agent take the step and gather the outputs
            next_obs, reward, terminated, truncated, res = env.step(action)

            # Use the next observation to get the next state (s')
            next_state = encode_state(next_obs)
            next_state_hash = env.hash(next_state)

            # Q-Value update
            if current_state_hash not in Q: 
                Q[current_state_hash] = numpy.zeros(NUM_ACTIONS)
            if next_state_hash not in Q: 
                Q[next_state_hash] = numpy.zeros(NUM_ACTIONS)

            Q[current_state_hash][action] += ALPHA * (reward + GAMMA * numpy.argmax(Q[next_state_hash] - Q[current_state_hash][action]))
    
            # Update the reward values for this episode
            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                steps_log.append(episode_steps)
                rewards_log.append(episode_reward)
                print("Episode {}: {}".format(episode + 1, "Success" if terminated else "Failed"))
                break

            # Episode is still going, get ready for the next step
            current_state = next_state

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
    env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
    env = SymbolicObsWrapper(env)
    qlearning()
    env.close()