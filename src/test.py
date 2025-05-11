# CS499_LavaCrossing/src/test.py
import gymnasium as gym
import minigrid

def list_minigrid_envs():
    from gymnasium.envs.registration import registry
    print("Available Minigrid environments:")
    minigrid_envs = [env_id for env_id in registry if env_id.startswith("MiniGrid-")]
    minigrid_envs.sort()
    for env_id in minigrid_envs:
        print(env_id)

def main():
    try:
        env_name = "MiniGrid-LavaCrossingS9N1-v0"
        print(f"Attempting to create environment: {env_name}")
        env = gym.make(env_name, render_mode="human")
        print(f"Successfully created environment: {env_name}")
        obs, _ = env.reset(seed=0)
        done, ep_ret = False, 0
        print("Starting episode...")
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward
            # print(f"Action: {action}, Reward: {reward}, Done: {done}")
        print("Episode finished.")
        print("Episode return:", ep_ret)
        env.close()
    except Exception as e:
        print(f"An error occurred during main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()