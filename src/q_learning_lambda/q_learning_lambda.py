import math
import numpy
import minigrid
import gymnasium as gym

from minigrid.wrappers import SymbolicObsWrapper
from minigrid.envs import crossing

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from encode_state import encode_state


# Initialize Q(s,a) arbitrarily and e(s,a) = 0, for all s,a
# Repeat (for each episode):
#     Initialize s, a
#     Repeat (for each step of episode):
#         Take action a, observe r, s'
#         Choose a' from s' using policy derived from Q (e.g., ε-greedy)
#         a* ← arg max_b Q(s',b) (if a' ties for the max, then a* ← a')
#         δ ← r + γQ(s',a*) - Q(s,a)
#         e(s,a) ← e(s,a) + 1
#         For all s,a:
#             Q(s,a) ← Q(s,a) + αδe(s,a)
#             If a' = a*, then e(s,a) ← γλe(s,a)
#                        else e(s,a) ← 0
#         s ← s'; a ← a'
#     until s is terminal


if __name__ == "__main__":
    env = gym.make("MiniGrid-LavaCrossingS9N1-v0", render_mode="human")
    env = SymbolicObsWrapper(env)

    obs, _ = env.reset(seed=42)
    print("state key =", encode_state(obs))

    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        key = encode_state(obs)
        print(f"Step {i+1} state key: {key}, reward: {reward}")
        if terminated or truncated:
            print("Episode ended.")
            break
    env.close()