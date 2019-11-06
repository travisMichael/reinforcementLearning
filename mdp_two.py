import gym
from CartPole import CartPoleEnv
# env = gym.make('CartPole-v0')
# state = env.reset()
env = CartPoleEnv()
env.reset()
state = env.state

next_state, reward, done, _ = env.step(0)
print(next_state, reward, done)

env.state = state

next_state, reward, done, _ = env.step(0)
print(next_state, reward, done)


env.close()