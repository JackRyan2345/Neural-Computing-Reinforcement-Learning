import gymnasium as gym

env = gym.make("Acrobot-v1", render_mode="rgb_array")  # rgb_array lets you record videos later
state, info = env.reset(seed=42)

print("What he sees")
print(env.observation_space)   # what the agent sees
print("What he can do")
print(env.action_space)        # what the agent can do