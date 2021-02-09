import os
import sys

import torch

from agent import MPPNAgent
from config.persistence import paths
from env import DAVIS2017Environment
from util import mkdir, convert_to_indexed, imwrite_indexed

CPU = torch.device('cpu')


def parse_args():
    try:
        model_type = sys.argv[1]  # str
        if model_type not in ('supervised', 'rl'):
            print('Invalid model type "%s", defaulting to "supervised".' % model_type)
            model_type = 'supervised'
        best_model_id = sys.argv[2]  # int
        device_id = sys.argv[3]  # str
    except IndexError:
        print('Usage:\tpython test.py <model_type> <best_model_id>')
        sys.exit()
    return model_type, int(best_model_id), 'cuda:' + device_id


def main():
    model_type, model_id, device = parse_args()
    env = DAVIS2017Environment('val', device)
    agent = MPPNAgent(device)
    agent.load_model(model_id, supervised=(model_type == 'supervised'))
    episode = 0
    while episode < len(env.data_loader):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, stochastic=False)
            next_state, reward, done, info = env.step(action)
            if next_state is not None:
                state = next_state

        save_path = os.path.join(paths[model_type]['masks'], 'val', info['seq_id'])
        mkdir(save_path)
        arr = env.total_proposal.to(CPU)
        arr = convert_to_indexed(arr).numpy()
        imwrite_indexed(os.path.join(save_path, info['img_id'] + '.png'), arr)
        print("Wrote output file for %s:%s" % (info['seq_id'], info['img_id']))

        episode += 1


if __name__ == '__main__':
    main()
