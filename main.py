import numpy as np
import torch
import gym

import TD3
import ReplayBuffer

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 0x123)

    avg_reward = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

        avg_reward /= eval_episodes
        print(20 * '-')
        print(f"Evaluation over {eval_episodes} episodes\tAverage reward: {avg_reward:.3f}")
        print(20 * '-')
        return avg_reward


if __name__ == "__main__":
    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)

    # Consts
    seed = 0                    # Sets gym, torch, np seeds
    exploration_noise = 0.1     # probability to choose action randomly
    timesteps = int(2e5)        # timesteps to run whole environment
    start_timesteps = 25e3      # initial timesteps - random policy is used
    batch_size = 256
    eval_freq = 5000

    save_model = True
    load_model = 'LunarLanderContinuous_Test'
    file_name = load_model

    # Set seeds
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3.TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim)

    if load_model != "":
        agent.load(f"./models/{load_model}")

    # Check untrained agent
    evals = [eval_policy(agent, env_name, seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Train - Eval loop
    for t in range(timesteps):
        episode_timesteps += 1

        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    agent.select_action(np.array(state))
                    + np.random.normal(0, max_action * exploration_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            agent.train(replay_buffer, batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} \
                Reward: {episode_reward:.3f}"
            )

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate agent
        if (t + 1) % eval_freq == 0:
            evals.append(eval_policy(agent, env_name, seed))
            np.save(f"./results/{file_name}", evals)
            if save_model:
                agent.save(f"./models/{file_name}")

