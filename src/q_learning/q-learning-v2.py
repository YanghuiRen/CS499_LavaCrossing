import math
import numpy
import matplot.pyplot as plt
import minigrid
import gymnasium as gym

from minigrid.wrappers import SymbolicObsWrapper
from minigrid.envs import crossing

env = gym.make("MiniGrid-LavaCrossingS11N5-v0")


# Initialize Q(s, a), ∀s ∈ S, a ∈ A(s), arbitrarily, and Q(terminal-state, ·) = 0
# Repeat (for each episode):
# Initialize S
# Repeat (for each step of episode):
# Choose A from S using policy derived from Q (e.g., epsilon-greedy)
# Take action A, observe R, S′
# Q(S, A) ← Q(S, A) + α[R + γ maxa Q(S′, a) − Q(S, A)]
# S ← S′;
# until S is terminal