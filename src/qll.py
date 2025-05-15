import header


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
def qlearn_lambda():
    ...










def main():
    """
    A little bit of testing to ensure that minigrid works. Can be deleted
    """
    env = header.gym.make("MiniGrid-LavaCrossingS11N5-v0", render_mode='human')
    obs, _ = env.reset()
    print(obs['image'].shape)
    
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        obs, r, done, truncated, info = env.step(action)

if __name__ == "__main__":
    main()