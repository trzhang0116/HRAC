import torch
import torch.nn as nn
import torch.utils.data as Data

import numpy as np


def train_anet(a_net, states, adj_mat, optimizer, margin_pos, margin_neg,
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
            if verbose and (j % 100 == 0 or j == n_batches - 1):
                print('Training adjacency network: epoch {}/{}, batch {}/{}'.format(i+1, n_epochs, j+1, n_batches))
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
        label = label.float()
        dist = torch.sqrt(torch.pow(x - y, 2).sum(dim=1) + 1e-12)
        loss = (label * (dist - self.margin_pos).clamp(min=0)).mean() \
             + ((1 - label) * (self.margin_neg - dist).clamp(min=0)).mean()
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
