import gym
import Learn

import sys
if "../" not in sys.path:
  sys.path.append("../")


class ValueIterationAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, env):
        return self.action_space.sample()


if __name__ == '__main__':
    outdir = '/tmp/results'
    env = gym.make('Taxi-v2')
    agent = ValueIterationAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    print env.action_space
    print env.observation_space

    # policy, v = Learn.value_iteration(env)
    #
    # print("Policy Probability Distribution:")
    # print(policy)
    # print("")

    # for i in range(episode_count):
    #     observation = env.reset()
    #
    #     for t in range(100):
    #         env.render()
    #         print(observation)
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t + 1))
    #             break

        # while True:
        #     env.render()
        #     action = agent.act(env)
        #     ob, reward, done, _ = env.step(action)
        #     if done:
        #         break
    #
        # env.step(env.action_space.sample()) # take a random action
