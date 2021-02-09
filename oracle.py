import json
import os
import sys
import time

import torch

from agent import Oracle
from config.persistence import paths
from env import DAVIS2017Environment
from util import mkdir, convert_to_indexed, imwrite_indexed

CPU = torch.device('cpu')


def parse_args():
    subset = sys.argv[1]  # str
    if subset not in ('train', 'val'):
        print('Invalid subset "%s", defaulting to "train".')
        subset = 'train'
    device_id = sys.argv[2]  # str
    return subset, 'cuda:' + device_id


def main(subset, device):
    env = DAVIS2017Environment(subset, device)
    with open('gt_{}.json'.format(subset), 'r') as fp:
        gt = json.load(fp)
    agent = Oracle(gt)
    episode = 0
    while episode < len(env.data_loader):
        start = time.time()
        _ = env.reset()
        done = False
        while not done:
            
            e = env.data_loader[env.episode_id]
            action = agent.act(e.seq_id, e.img_id, env.state_id)
            next_state, reward, done, info = env.step(action, oracle=True)
            

        save_path = os.path.join(paths['oracle']['masks'], subset, info['seq_id'])
        mkdir(save_path)
        arr = env.total_proposal.to(CPU)
        arr = convert_to_indexed(arr).numpy()
        imwrite_indexed(os.path.join(save_path, info['img_id'] + '.png'), arr)
        print((time.time() - start))

        print("Wrote output file for %s:%s" % (info['seq_id'], info['img_id']))

        episode += 1


if __name__ == '__main__':
    subset, device = parse_args()
    main(subset, device)
