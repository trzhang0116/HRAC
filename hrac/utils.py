import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, maxsize=1e6):
        self.storage = [[] for _ in range(8)]
        self.maxsize = maxsize
        self.next_idx = 0

    def clear(self):
        self.storage = [[] for _ in range(8)]
        self.next_idx = 0

    # Expects tuples of (x, x', g, u, r, d, x_seq, a_seq)
    def add(self, data):
        self.next_idx = int(self.next_idx)
        if self.next_idx >= len(self.storage[0]):
            [array.append(datapoint) for array, datapoint in zip(self.storage, data)]
        else:
            [array.__setitem__(self.next_idx, datapoint) for array, datapoint in zip(self.storage, data)]

        self.next_idx = (self.next_idx + 1) % self.maxsize

    def sample(self, batch_size):
        if len(self.storage[0]) <= batch_size:
            ind = np.arange(len(self.storage[0]))
        else:
            ind = np.random.randint(0, len(self.storage[0]), size=batch_size)

        x, y, g, u, r, d, x_seq, a_seq = [], [], [], [], [], [], [], []          

        for i in ind: 
            X, Y, G, U, R, D, obs_seq, acts = (array[i] for array in self.storage)
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            g.append(np.array(G, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

            # For off-policy goal correction
            x_seq.append(np.array(obs_seq, copy=False))
            a_seq.append(np.array(acts, copy=False))
        
        return np.array(x), np.array(y), np.array(g), \
            np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), \
            x_seq, a_seq

    def save(self, file):
        np.savez_compressed(file, idx=np.array([self.next_idx]), x=self.storage[0],
                            y=self.storage[1], g=self.storage[2], u=self.storage[3],
                            r=self.storage[4], d=self.storage[5], xseq=self.storage[6],
                            aseq=self.storage[7])

    def load(self, file):
        with np.load(file) as data:
            self.next_idx = int(data['idx'][0])
            self.storage = [data['x'], data['y'], data['g'], data['u'], data['r'],
                            data['d'], data['xseq'], data['aseq']]
            self.storage = [list(l) for l in self.storage]

    def __len__(self):
        return len(self.storage[0])


class TrajectoryBuffer(object):

    def __init__(self, capacity):
        self._capacity = capacity
        self.reset()

    def reset(self):
        self._num_traj = 0  # number of trajectories
        self._size = 0    # number of game frames
        self.trajectory = []

    def __len__(self):
        return self._num_traj

    def size(self):
        return self._size

    def get_traj_num(self):
        return self._num_traj

    def full(self):
        return self._size >= self._capacity

    def create_new_trajectory(self):
        self.trajectory.append([])
        self._num_traj += 1

    def append(self, s):
        self.trajectory[self._num_traj-1].append(s)
        self._size += 1

    def get_trajectory(self):
        return self.trajectory

    def set_capacity(self, new_capacity):
        assert self._size <= new_capacity
        self._capacity = new_capacity


class NormalNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def perturb_action(self, action, min_action=-np.inf, max_action=np.inf):
        action = (action + np.random.normal(0, self.sigma,
            size=action.shape)).clip(min_action, max_action)
        return action


class OUNoise(object):
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.3):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def perturb_action(self, action, min_action=-np.inf, max_action=np.inf):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return (self.X + action).clip(min_action, max_action)


def train_adj_net(a_net, states, adj_mat, optimizer, margin_pos, margin_neg,
                  n_epochs=100, batch_size=64, device='cpu', verbose=False):
    if verbose:
        print('Generating training data...')
    dataset = MetricDataset(states, adj_mat)
    if verbose:
        print('Totally {} training pairs.'.format(len(dataset)))
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    n_batches = len(dataloader)

    loss_func = ContrastiveLoss(margin_pos, margin_neg)

    for i in range(n_epochs):
        epoch_loss = []
        for j, data in enumerate(dataloader):
            x, y, label = data
            x = x.float().to(device)
            y = y.float().to(device)
            label = label.long().to(device)
            x = a_net(x)
            y = a_net(y)
            loss = loss_func(x, y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose and (j % 50 == 0 or j == n_batches - 1):
                print('Training metric network: epoch {}/{}, batch {}/{}'.format(i+1, n_epochs, j+1, n_batches))

            epoch_loss.append(loss.item())

        if verbose:
            print('Mean loss: {:.4f}'.format(np.mean(epoch_loss)))


class ContrastiveLoss(nn.Module):

    def __init__(self, margin_pos, margin_neg):
        super().__init__()
        assert margin_pos <= margin_neg
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg

    def forward(self, x, y, label):
        # mutually reachable states correspond to label = 1
        dist = torch.sqrt(torch.pow(x - y, 2).sum(dim=1) + 1e-12)
        loss = (label * (dist - self.margin_pos).clamp(min=0)).mean() + ((1 - label) * (self.margin_neg - dist).clamp(min=0)).mean()
        return loss


class MetricDataset(Data.Dataset):

    def __init__(self, states, adj_mat):
        super().__init__()
        n_samples = adj_mat.shape[0]
        self.x = []
        self.y = []
        self.label = []
        for i in range(n_samples - 1):
            for j in range(i + 1, n_samples):
                self.x.append(states[i])
                self.y.append(states[j])
                self.label.append(adj_mat[i, j])
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.label = np.array(self.label)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.label[idx]
