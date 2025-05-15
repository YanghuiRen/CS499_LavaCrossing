from minigrid.core.constants import OBJECT_TO_IDX
import numpy as np
import gymnasium as gym
from minigrid.wrappers import SymbolicObsWrapper

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



if __name__ == "__main__":
    env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
    env = SymbolicObsWrapper(env)
    obs, _ = env.reset(seed=42)