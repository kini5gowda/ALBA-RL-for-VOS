import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as f
from torch.distributions import Categorical

from config.persistence import paths
from model import SelectionNetwork as MPPN


class Oracle(object):
    def __init__(self, gt):
        self.gt = gt

    def act(self, seq_id, img_id, state_id):
        return self.gt[seq_id][img_id][state_id]


class MPPNAgent(object):
    def __init__(self, device, gamma=0.99):
        self.gamma = gamma
        self.eps = np.finfo(np.float32).eps.item()
        self.model = MPPN()
        self.device = device
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.logprobs = []
        self.rewards = []

    def train(self):
        r = self.compute_discounted_rewards().to(self.device)
        logprobs = torch.stack(self.logprobs)
        self.optimizer.zero_grad()
        loss = (-logprobs * r).sum()
        loss.backward()
        self.optimizer.step()
        self.logprobs = []
        self.rewards = []

    def act(self, state, stochastic=True):
        probs = f.softmax(self.model(state)[0], dim=0)
        if stochastic:
            b = Categorical(probs)
            action = b.sample()
            self.logprobs.append(b.log_prob(action))
        else:
            action = probs.argmax()
        return action

    def compute_discounted_rewards(self):
        r = np.array(
            [self.gamma**i * self.rewards[i] for i in range(len(self.rewards))])
        r = r[::-1].cumsum()[::-1]
        r = (r - r.mean()) / (r.std() + self.eps)
        return torch.from_numpy(r)

    def save_model(self, episode):
        path = paths['rl']['models.select']
        filename = 'ep{:06d}.pt'.format(episode)
        torch.save(self.model.state_dict(), path + filename)

    def load_model(self, model_id, supervised=True):
        if supervised:
            path = paths['supervised']['models.select']
            print(path)
            filename = 'ep{:03d}.pt'.format(model_id)
        else:
            path = paths['rl']['models.select']
            filename = 'ep{:06d}.pt'.format(model_id)
        self.model.load_state_dict(torch.load(path + filename))
