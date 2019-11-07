import gym
# from CartPole import CartPoleEnv

env = gym.make('MountainCar-v0')
env.seed(505)
# env = gym.make('CartPole-v0')
# state = env.reset()
# env = CartPoleEnv()
env.reset()
# state = env.state

print(env.observation_space.low)
print('--')
print(env.observation_space.high)
print('--')
print(env.action_space.n)
print('--')

next_state, reward, done, _ = env.step(0)
print(next_state, reward, done)

# env.state = state

next_state, reward, done, _ = env.step(0)
print(next_state, reward, done)

env.close()
