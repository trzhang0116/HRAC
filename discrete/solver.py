import torch
import numpy as np

from agent import Agent


class Solver:

    def __init__(self, args):
        self.set_seed(args.seed)
        self.agent = Agent(args)

    def set_seed(self, seed):
        # set random seed for reproducibility
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

    def train(self):
        self.agent.train()

    def show(self):
        self.agent.show()
