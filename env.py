import torch
import torch.nn.functional as f

from config.model import max_objects
from data_loader import DAVIS2017Loader
from metrics import IoU, IoU_multi_obj


def standardize(x):
    return (x - x.mean()) / x.std()


class DAVIS2017Environment(object):
    def __init__(self, subset, device):
        if subset not in ('train', 'val'):
            raise ValueError('Subset must be "train" or "val".')
        self.subset = subset
        self.episode_id = -1
        self.data_loader = DAVIS2017Loader(self.subset, device)
        self.device = device
        # Initialized in reset.
        self.current_episode = None
        self.current_state = None
        self.state_id = None
        self.total_proposal = None
        self.action_idx = None
        self.total_reward = None

    def _increment_episode(self):
        self.episode_id += 1
        if self.episode_id >= len(self.data_loader):
            self.episode_id = 0

    def reset(self):
        self._increment_episode()
        self.current_episode = self.data_loader[self.episode_id]
        num_proposals, H, W = self.current_episode.mask_proposals.shape
        self.total_proposal = torch.zeros(max_objects, H, W).to(self.device)
        self.action_idx = 0
        self.total_reward = 0.
        # Return the first state of the episode.
        self.state_id = 0
        self.current_state = self.get_state()
        return self.current_state

    def get_state(self):
        state = {}
        mask_proposals = self.current_episode.mask_proposals
        mask_features = self.current_episode.mask_features
        num_proposals, H, W = mask_proposals.shape
        optical_flow = standardize(self.current_episode.optical_flow.reshape(2, H, W))

        state['flow_features'] = f.interpolate(torch.cat([
            mask_proposals[self.state_id].float().unsqueeze(0),
            optical_flow
        ]).unsqueeze(0), size=(256, 256), mode='nearest')
        state['visual_features'] = mask_features[self.state_id].mean(2).mean(1).unsqueeze(0).to(self.device)

        return state

    def step(self, action, oracle=False):
        info = {
            'seq_id': self.current_episode.seq_id,
            'img_id': self.current_episode.img_id
        }

        p = self.current_episode.mask_proposals[self.state_id]
        # The oracle tells us which total proposal index to use.
        if oracle:
            if action != 0:
                self.total_proposal[action-1] = torch.clamp(
                    self.total_proposal[action-1] + p, 0, 1)
        else:
            if action != 0 and self.action_idx < max_objects:
                self.total_proposal[self.action_idx] = self.current_episode.mask_proposals[self.state_id]
                self.action_idx += 1

        reward = IoU_multi_obj(self.total_proposal, self.current_episode.annotation).item()
        true_reward = reward - self.total_reward
        self.total_reward = reward

        self.state_id += 1
        try:
            next_state = self.get_state()
            done = False
        except IndexError:
            next_state = self.current_state
            done = True
        self.current_state = next_state

        return next_state, true_reward, done, info
