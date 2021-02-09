import sys

import numpy as np
from datetime import datetime

from agent import MPPNAgent
from config.persistence import paths
from env import DAVIS2017Environment


def parse_args():
    try:
        num_episodes = sys.argv[1]  # int
        init_model_id = sys.argv[2]  # int
        device_id = sys.argv[3]  # str
    except IndexError:
        print('Usage:\tpython train.py <num_episodes> <init_model_id> <device_id>')
        sys.exit()
    return int(num_episodes), int(init_model_id), 'cuda:' + device_id


def main(num_episodes, init_model_id, device):
    env = DAVIS2017Environment('train', device)
    agent = MPPNAgent(device)
    if init_model_id:
        print('Loading pre-trained supervised model from epoch %d.' % (init_model_id + 1))
        agent.load_model(init_model_id, supervised=True)

    end_rewards = []
    reward_moving_avg = []
    window_size = len(env.data_loader)
    print('Moving average window size: %d' % window_size)

    episode = 0
    while episode < num_episodes:
        ts_start = datetime.now()

        steps = 0
        done = False
        state = env.reset()
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.rewards.append(reward)
            steps += 1
            state = next_state
        agent.train()

        end_reward = env.total_reward
        end_rewards.append(end_reward)
        reward_moving_avg.append(end_reward)
        if len(reward_moving_avg) > window_size:
            reward_moving_avg = reward_moving_avg[1:]

        if len(end_rewards) == 1000:
            save_path = paths['rl']['rewards']
            filename = 'ep{:06d}.pt'.format(episode)
            np.save(save_path + filename, np.array(end_rewards))
            end_rewards = []

        if (len(reward_moving_avg)) == window_size and (np.mean(reward_moving_avg) > 0.65):
            agent.save_model(episode)

        ts_end = datetime.now()
        episode_interval = (ts_end - ts_start).total_seconds()
        print("Episode {} ({}:{}) ended in {} seconds. Final reward: {}".format(
            episode + 1, info['seq_id'], info['img_id'], episode_interval, end_reward))

        episode += 1


if __name__ == '__main__':
    num_episodes, init_model_id, device = parse_args()
    main(num_episodes, init_model_id, device)
