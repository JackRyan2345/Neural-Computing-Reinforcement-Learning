import gymnasium as gym
import numpy as np

env = gym.make("Acrobot-v1")
returns = []
for ep in range(20):
    state, info = env.reset(seed=ep)
    total, done = 0, False
    while not done:
        action = env.action_space.sample()                     # random action
        state, reward, terminated, truncated, info = env.step(action)
        total += reward
        done = terminated or truncated
    returns.append(total)

print(f"Random policy: {np.mean(returns):.1f} ± {np.std(returns):.1f}")