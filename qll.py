import gymnasium as gym

from minigrid.wrappers import SymbolicObsWrapper
# from minigrid.core.constants import COLOR_NAMES
# from minigrid.core.grid import Grid
# from minigrid.core.mission import MissionSpace
# from minigrid.core.world_object import Door, Goal, Key, Wall
# from minigrid.manual_control import ManualControl
# from minigrid.minigrid_env import MiniGridEnv


def main():
    """
    A little bit of testing to ensure that minigrid works. Can be deleted
    """
    env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
    obs, _ = env.reset()
    print(obs['image'].shape)
    
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()





"""
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
"""





if __name__ == "__main__":
    main()