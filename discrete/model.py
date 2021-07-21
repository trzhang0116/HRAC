import torch
import torch.nn as nn
import torch.nn.functional as F

import copy


class HRLNet(nn.Module):

    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim, x_range, y_range, horizon,
        n_noisy_goals, soft_sync_rate, low_action_scale=None):
        super().__init__()
        self.goal_dim = goal_dim
        self.soft_sync_rate = soft_sync_rate
        self.low_action_scale = low_action_scale

        self.actor_h = HighLevelActorNet(state_dim, goal_dim, hidden_dim, x_range, y_range, horizon, n_noisy_goals)
        self.critic_h = CriticNet(state_dim, goal_dim, hidden_dim)
        self.actor_h_tgt = copy.deepcopy(self.actor_h)
        self.critic_h_tgt = copy.deepcopy(self.critic_h)

        self.policy_l = LowLevelPolicyNet(state_dim, goal_dim, action_dim, hidden_dim)
        
    def soft_sync_high(self):
        self._soft_sync(self.actor_h, self.actor_h_tgt)
        self._soft_sync(self.critic_h, self.critic_h_tgt)

    def _soft_sync(self, net, tgt_net):
        for param, tgt_param in zip(net.parameters(), tgt_net.parameters()):
            tgt_param.data.copy_(self.soft_sync_rate * param.data + (1 - self.soft_sync_rate) * tgt_param.data)

    def goal_transition(self, state, last_state, last_goal):
        return last_goal

    def forward(self, state, last_state, last_goal, flag, explore_sigma_high, evaluate=False):
        if flag:
            goal = self.actor_h(state, explore_sigma_high, evaluate=evaluate)
        else:
            goal = self.goal_transition(state, last_state, last_goal)

        policy, value = self.policy_l(state, goal)
        prob = F.softmax(policy, dim=-1)
        log_prob = F.log_softmax(policy, dim=-1)
        return goal, prob, log_prob, value


class ANet(nn.Module):

    def __init__(self, goal_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(goal_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class HighLevelActorNet(nn.Module):
    # TD3 Actor
    def __init__(self, state_dim, goal_dim, hidden_dim, x_range, y_range, horizon, n_noisy_goals):
        super().__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.horizon = horizon
        self.n_noisy_goals = n_noisy_goals

        self.goal_fc = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, goal_dim))

    def forward(self, state, explore_sigma, evaluate=False):
        goal = torch.sigmoid(self.goal_fc(state))
        noise_flag = self.n_noisy_goals > 0 and goal.size(0) == 1 and not evaluate
        if noise_flag:
            raw_goal = goal
            goal = goal.expand(self.n_noisy_goals, goal.size(1))
        noise = explore_sigma * torch.randn(goal.size()).to(goal.device)
        scale = torch.FloatTensor([self.x_range, self.y_range]).expand_as(goal).to(goal.device)
        goal = goal * scale
        if evaluate:
            output = goal
        else:
            output = torch.min(torch.clamp(goal + noise, min=0), scale)
            if noise_flag:
                output = torch.cat((output, raw_goal), dim=0)
        return output


class CriticNet(nn.Module):
    # TD3 Critic
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.value_fc_1 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, 1))

        self.value_fc_2 = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        return self.value_fc_1(x).squeeze(-1), self.value_fc_2(x).squeeze(-1)

    def value(self, state, action):
        return self.value_fc_1(torch.cat((state, action), dim=-1)).squeeze(-1)


class LowLevelPolicyNet(nn.Module):
    # Low-level A2C
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim):
        super().__init__()
        self.action_fc = nn.Sequential(nn.Linear(state_dim + goal_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, action_dim))

        self.value_fc = nn.Sequential(nn.Linear(state_dim + goal_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, goal):
        goal = goal.detach()  # no gradient
        x = torch.cat((state, goal), dim=-1)
        policy = self.action_fc(x)
        value = self.value_fc(x).squeeze(-1)
        return policy, value
