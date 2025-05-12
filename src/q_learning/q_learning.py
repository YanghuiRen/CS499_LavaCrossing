import numpy as np, matplotlib.pyplot as plt, gymnasium as gym, minigrid
from minigrid.wrappers import FullyObsWrapper
from collections import defaultdict
from pathlib import Path
from datetime import datetime

OBJ_AGENT, OBJ_GOAL = 10, 8
def first_pos(img, idx):
    loc = np.argwhere(img[:, :, 0] == idx)
    return tuple(loc[0]) if len(loc) else (-1, -1)
def encode_state(img):
    return first_pos(img, OBJ_AGENT) + first_pos(img, OBJ_GOAL)

ACTION_MAP = {0: 0, 1: 1, 2: 2}
N_AGENT_ACT = len(ACTION_MAP)

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.9995):
        self.gamma, self.alpha = gamma, alpha
        self.epsilon = eps_start
        self.eps_min, self.decay = eps_end, eps_decay
        self.Q = defaultdict(lambda: np.zeros(N_AGENT_ACT, dtype=np.float32))

    def get_action(self, state, rng):
        if rng.random() < self.epsilon:
            return rng.integers(N_AGENT_ACT)
        qs = [self.Q[(state, a)] for a in range(N_AGENT_ACT)]
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state, done):
        next_q_max = 0 if done else self.Q[next_state].max()
        td_target  = reward + self.gamma * next_q_max
        self.Q[state][action] += self.alpha * (td_target - self.Q[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.decay, self.eps_min)

def run_trial(alpha, seed, episodes=2000):
    rng = np.random.default_rng(seed)
    env = FullyObsWrapper(gym.make("MiniGrid-LavaCrossingS9N1-v0"))
    agent = QLearningAgent(alpha=alpha)
    rewards = np.zeros(episodes)

    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(1e9)))
        state = encode_state(obs["image"])
        done, ep_ret = False, 0.0

        while not done:
            a_idx = agent.get_action(state, rng)
            env_action = ACTION_MAP[a_idx]
            obs2, r, term, trunc, _ = env.step(env_action)
            done = term or trunc
            next_state = encode_state(obs2["image"])
            agent.update(state, a_idx, r, next_state, done)
            state, ep_ret = next_state, ep_ret + r

        rewards[ep] = ep_ret
        agent.decay_epsilon()

    env.close()
    return rewards


# Save results and generate images
'''
timestamp = datetime.now().strftime("%b%d_%H%M")
results_dir = f"results/{timestamp}_alpha_compare"
Path(results_dir).mkdir(parents=True, exist_ok=True)
Path("images").mkdir(exist_ok=True)

curves = {}
for alpha in (0.1, 0.5):
    curves[alpha] = np.mean([run_trial(alpha, s) for s in range(5)], axis=0)
    np.save(f"{results_dir}/reward_alpha{alpha}.npy", curves[alpha])

win = 100
avg = lambda x: np.convolve(x, np.ones(win)/win, mode="valid")
x = np.arange(win-1, len(curves[0.1]))

plt.figure(figsize=(9,5))
plt.plot(x, avg(curves[0.1]), label="α = 0.1", lw=2)
plt.plot(x, avg(curves[0.5]), label="α = 0.5", lw=2)
plt.xlabel("Episode"); plt.ylabel(f"Average reward ({win}-episode)")
plt.title("Q-Learning α Comparison (5 trials)")
plt.legend(); plt.tight_layout()

plt.savefig("images/q_learning_alpha_compare.png", dpi=300)
print("Saved figure → images/q_learning_alpha_compare.png")
print(f"Saved data → {results_dir}/")
'''
