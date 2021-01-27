import numpy as np
import gym
import TD3

if __name__ == '__main__':
    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    file_name = 'LunarLanderContinuous-v2_Plot'
    agent = TD3.TD3(state_dim, action_dim, max_action)
    if file_name != "":
        agent.load(f"./models/{file_name}")

    episodes = 10
    avg_reward = 0
    for e in range(episodes):
        state, done = env.reset(), False
        episode_time_steps = 0
        episode_reward = 0
        while not done:
            env.render()

            # Agent action
            action = agent.select_action(np.array(state))
            state, reward, done, _ = env.step(action)

            # Random action
            # state, reward, done, _ = env.step(env.action_space.sample())

            episode_reward += reward
            episode_time_steps += 1

        input()
        avg_reward += episode_reward
        print(f"Episode {e} \t Reward: {episode_reward:.3f} \t Timesteps: {episode_time_steps:.3f}")

    avg_reward /= episodes
    print(20 * '-')
    print(f"Evaluation over {episodes} episodes\tAverage reward: {avg_reward:.3f}")
    print(20 * '-')
    env.close()
