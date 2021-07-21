import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import time
import datetime
import json
import numpy as np
import pandas as pd

from env import MazeEnv, KeyChestEnv
from model import HRLNet, ANet
from memory import HighLevelReplayBuffer, LowLevelMemory, TrajectoryMemory
from metric import train_anet
import utils


class Agent:

    def __init__(self, args):
        self.algo = args.algo
        self.k = args.manager_propose_freq
        self.goal_loss_coeff = args.goal_loss_coeff
        self.n_noisy_goals = args.n_noisy_goals
        self.man_discount = args.man_discount
        self.ctrl_discount = args.ctrl_discount

        self.r_margin_pos = args.r_margin_pos
        self.r_margin_neg = args.r_margin_neg
        self.r_init_epochs = args.r_init_epochs
        self.r_training_epochs = args.r_training_epochs
        self.r_training_freq = args.r_training_freq
        self.r_batch_size = args.r_batch_size

        self.man_noise_sigma = args.man_noise_sigma
        self.man_policy_update_freq = args.man_policy_update_freq

        self.ctrl_entropy = args.ctrl_entropy

        self.man_rew_scale = args.man_rew_scale
        self.ctrl_rew_scale = args.ctrl_rew_scale

        self.eval_freq = args.eval_freq
        self.eval_episodes = args.eval_episodes
        self.save_models = args.save_models
        self.model_save_freq = 10000
        self.log_freq = 1

        self.device = torch.device('cuda:{}'.format(args.gid))

        if args.env_name == 'Maze':
            self.env = MazeEnv(step_limit=200, seed=args.seed)
            self.eval_env = MazeEnv(step_limit=200, seed=args.seed+100)
            self.x_range = self.env.h
            self.y_range = self.env.w
            self.action_scale_l = None
            self.total_training_frames = 1000000
            self.random_start = False
        elif args.env_name == 'KeyChest':
            self.env = KeyChestEnv(step_limit=500, random_start=True, seed=args.seed)
            self.eval_env = KeyChestEnv(step_limit=500, random_start=True, seed=args.seed+100)
            self.x_range = self.env.h
            self.y_range = self.env.w
            self.action_scale_l = None
            self.total_training_frames = 2000000
            self.random_start = True
        else:
            raise NotImplementedError

        self.obs_dim = self.env.obs_dim
        self.action_dim = self.env.action_dim
        self.goal_dim = 2

        self.origin_time = datetime.datetime.now().strftime('%m%d-%H%M')
        output_path = os.path.join('output', args.env_name, self.origin_time)
        self.model_load_path = os.path.join('trained_models', args.env_name)
        self.model_save_path = os.path.join(output_path, 'models')
        self.log_path = os.path.join(output_path, 'log')
        self.result_path = os.path.join(output_path, 'results')
        self.output_data = {'frames': [], 'reward': []}
        self.output_filename = '{}_{}_{}.csv'.format(args.env_name, self.algo, args.seed)

        if self.save_models:
            utils.make_path(self.model_save_path)
        utils.make_path(self.log_path)
        utils.make_path(self.result_path)
        self.summary_writer = SummaryWriter(self.log_path)
        with open(os.path.join(self.result_path, 'params.json'), 'w') as json_file:
            json.dump(vars(args), json_file, indent=0)

        self.replay_buffer_h = HighLevelReplayBuffer(args.man_buffer_size, args.man_batch_size,
            self.obs_dim, self.goal_dim)
        self.memory_l = LowLevelMemory()

        self.build_model(args)
        if args.load_model:
            self.load_model()

        self.n_states = 0
        self.state_list = []
        self.state_dict = {}
        self.adj_mat = np.diag(np.ones(500, dtype=np.uint8))
        self.traj_memory = TrajectoryMemory(capacity=args.r_init_steps)

        self._episodes = 0
        self._frames = 0
        self.policy_update_it = 0
        self.loss_l = None
        self.loss_h = None
        self.mean_int_reward = None

    def build_model(self, args):
        self.net = HRLNet(self.obs_dim, self.goal_dim, self.action_dim, args.hidden_dim,
            self.x_range, self.y_range, self.k, self.n_noisy_goals, args.man_soft_sync_rate,
            low_action_scale=self.action_scale_l)
        self.net.to(self.device)

        self.a_net = ANet(self.goal_dim, args.r_hidden_dim, args.r_embedding_dim)
        self.a_net.to(self.device)

        self.optimizer_r = optim.Adam(self.a_net.parameters(), lr=args.lr_r)
        self.optimizer_actor_h = optim.Adam(self.net.actor_h.parameters(), lr=args.man_act_lr)
        self.optimizer_critic_h = optim.Adam(self.net.critic_h.parameters(), lr=args.man_crit_lr, weight_decay=args.weight_decay_critic)

        self.optimizer_l = optim.Adam(self.net.policy_l.parameters(), lr=args.ctrl_lr)

    def load_model(self):
        filename = os.path.join(self.model_load_path, '{}.pth'.format(self.algo))
        print('Loading the trained model: {}...'.format(filename))
        self.net.load_state_dict(torch.load(filename))

    def save_model(self, episode=None):
        if episode is not None:
            filename = os.path.join(self.model_save_path, '{}_{}.pth'.format(self.algo, episode))
        else:
            filename = os.path.join(self.model_save_path, '{}.pth'.format(self.algo))
        torch.save(self.net.state_dict(), filename)
        print('************** Model {} saved. **************'.format(episode))

    def train(self):
        print("===================== Training {} starts =====================".format(self.algo))
        self.start_time = time.time()

        print('Pre-training adjacency network...')
        self.env.pure_exploration = True
        self.env.random_start = True
        while not self.traj_memory.full():
            self._episodes += 1
            self.interact_one_episode(train=False)
            # print('Gathered samples: {} / {}'.format(self.traj_memory.size(), self.traj_memory._capacity))
        self.update_adj_mat()
        train_anet(self.a_net, self.state_list, self.adj_mat[:self.n_states, :self.n_states], self.optimizer_r,
                   self.r_margin_pos, self.r_margin_neg, n_epochs=self.r_init_epochs, batch_size=self.r_batch_size,
                   device=self.device, verbose=False)
        self.test_adjacency_acc()
        self.env.pure_exploration = False
        self.env.random_start = self.random_start

        self.traj_memory.reset()
        self.traj_memory.set_capacity(self.r_training_freq)

        while self._frames <= self.total_training_frames:
            if self._episodes == 0:
                self.test()
            self.interact_one_episode()
            self.train_one_episode()
            self._episodes += 1
            if self.traj_memory.full():
                print('Training adjacency network...')
                self.update_adj_mat()
                train_anet(self.a_net, self.state_list, self.adj_mat[:self.n_states, :self.n_states], self.optimizer_r,
                           self.r_margin_pos, self.r_margin_neg, n_epochs=self.r_training_epochs,
                           batch_size=self.r_batch_size, device=self.device, verbose=False)
                self.test_adjacency_acc()
                self.traj_memory.reset()

            if self._episodes % self.log_freq == 0:
                self.log_train()
            if self._episodes % self.eval_freq == 0:
                self.test()
            if self._episodes % self.model_save_freq == 0 and self.save_models:
                self.save_model(self._episodes)

        if self.save_models:
            self.save_model('last')
            r_filename = os.path.join(self.model_save_path, 'a_network.pth'.format(self._episodes))
            torch.save(self.a_net.state_dict(), r_filename)
        self.summary_writer.close()
        output_df = pd.DataFrame(self.output_data)
        output_df.to_csv(os.path.join(self.result_path, self.output_filename), float_format='%.4f', index=False)

        print("======================= Training {} ends =======================".format(self.algo))

    def test(self):
        print("[@@ {} @@] ************** Testing at episode {} **************".format(self.algo, self._episodes))
        reward_total = 0.
        dist_total = 0.
        for i in range(self.eval_episodes):
            reward, dist = self.test_one_episode()
            reward_total += reward
            dist_total += dist
        reward_avg = reward_total / self.eval_episodes
        dist_avg = dist_total / self.eval_episodes
        print('Average reward: {:.4f}, average dist: {:.4f}'.format(reward_avg, dist_avg))
        self.output_data['frames'].append(self._frames)
        self.output_data['reward'].append(reward_avg)
        # self.output_data['dist'].append(dist_avg)
        self.summary_writer.add_scalar('average test reward', reward_avg, self._frames)
        self.summary_writer.add_scalar('average test dist', dist_avg, self._frames)

    def interact_one_episode(self, train=True):
        self.env.new_episode()
        obs_curr = self.env.get_state()
        self.memory_l.reset()
        self.traj_memory.create_new_trajectory()
        last_state = None
        last_goal = None

        if train:
            r_horizon = 0.
            start_flag = self.replay_buffer_h.start()
            while True:
                obs_var = utils.single_input_transform(obs_curr, device=self.device)
                flag = (self.env.get_current_step() % self.k == 0)

                action, last_goal, last_state, (prob, log_prob, log_prob_act, value_l) = self.step(
                    obs_var, last_state, last_goal, flag, start_flag=start_flag)

                action_copy = action.copy()
                r = self.man_rew_scale * self.env.make_action(action_copy)
                obs_new = self.env.get_state()
                done = self.env.is_episode_finished()
                last_goal_np = last_goal.detach().squeeze().cpu().numpy()

                if (self.env.get_current_step() - 1) % self.k != 0:
                    r_horizon += r
                else:
                    if (self.env.get_current_step() - 1) != 0:
                        self.replay_buffer_h.append(state_store, goal_store, r_horizon, obs_curr, done)
                    goal_store = last_goal.detach().squeeze().cpu().numpy()
                    state_store = obs_curr
                    r_horizon = r

                info_low = dict(prob=prob, log_prob=log_prob, log_prob_act=log_prob_act, value_l=value_l)
                self.memory_l.append(last_state, last_goal, action, info_low)
                self.traj_memory.append(obs_curr)
                obs_curr = obs_new

                if done:
                    self.replay_buffer_h.append(state_store, goal_store, r_horizon, obs_curr, done)
                    obs_var = utils.single_input_transform(obs_curr, device=self.device)
                    self.memory_l.states.append(obs_var)
                    self.traj_memory.append(obs_curr)
                    self._frames += self.env.get_current_step()
                    self.curr_reward = self.env.get_total_reward()
                    break
        else:
            while True:
                obs_var = utils.single_input_transform(obs_curr, device=self.device)
                flag = (self.env.get_current_step() % self.k == 0)
                self.traj_memory.append(obs_curr)

                action, last_goal, last_state, _ = self.step(
                    obs_var, last_state, last_goal, flag, start_flag=False, evaluate=False)
                self.env.make_action(action)
                obs_new = self.env.get_state()
                done = self.env.is_episode_finished()
                obs_curr = obs_new
                if done:
                    self.traj_memory.append(obs_curr)
                    self._frames += self.env.get_current_step()
                    break

    def step(self, state, last_state, last_goal, flag, start_flag=True, evaluate=False):
        if evaluate:
            goal, prob, log_prob, value_l = self.net(
                state, last_state, last_goal, flag,
                self.man_noise_sigma, evaluate=True)
            action = prob.multinomial(1)
            log_prob_act = log_prob.gather(1, action).squeeze()
        else:
            if not start_flag:
                if flag:
                    goal = torch.rand(self.goal_dim).unsqueeze(0).to(self.device)
                    scale = torch.FloatTensor([self.x_range, self.y_range]).expand_as(goal).to(self.device)
                    goal = goal * scale
                else:
                    goal = self.goal_transition(state, last_state, last_goal)
                policy, value_l = self.net.policy_l(state, goal)
                prob = F.softmax(policy, dim=-1)
                log_prob = F.log_softmax(policy, dim=-1)
                action = prob.multinomial(1)
                log_prob_act = log_prob.gather(1, action).squeeze()
            else:
                if flag:
                    goals = self.net.actor_h(state, self.man_noise_sigma, evaluate=False)
                    if self.n_noisy_goals > 0:
                        noised_goals = goals[:self.n_noisy_goals]
                        raw_goal = goals[-1].unsqueeze(0)
                        goal = self.sample_adjacent_subgoal(noised_goals, state)
                        if goal is None:
                            goal = raw_goal
                    else:
                        goal = goals
                else:
                    goal = self.goal_transition(state, last_state, last_goal)
                policy, value_l = self.net.policy_l(state, goal)
                prob = F.softmax(policy, dim=-1)
                log_prob = F.log_softmax(policy, dim=-1)
                action = prob.multinomial(1)
                log_prob_act = log_prob.gather(1, action).squeeze()
        return action.squeeze().detach().cpu().numpy(), goal, state, (prob, log_prob, log_prob_act, value_l)

    def sample_adjacent_subgoal(self, goals, state):
        # Randomly sample an adjacent subgoal from a goal list
        inputs = torch.cat((state[:, :self.goal_dim], goals), dim=0)
        outputs = self.a_net(inputs)
        s_embedding = outputs[0].unsqueeze(0)
        goal_embeddings = outputs[1:]
        dists = utils.euclidean_dist(s_embedding, goal_embeddings).squeeze()
        idx = (dists < (self.r_margin_neg + self.r_margin_pos) / 2).nonzero().squeeze()
        if idx.size() and len(idx) == 0:
            return None
        elif not idx.size():  # one index
            sel_goal = goals[idx]
        else:
            sample_idx = np.random.randint(len(idx))
            sel_goal = goals[idx[sample_idx]]
        sel_goal = sel_goal.unsqueeze(0)
        return sel_goal

    def goal_transition(self, state, last_state, last_goal):
        return last_goal

    def train_one_episode(self):
        # train low level
        if self.replay_buffer_h.start() and self.memory_l.size() > 0:
            self.loss_policy_l, self.loss_value_l, self.loss_entropy_l = self.train_low_level_a2c()
            self.loss_l = self.loss_policy_l + self.loss_value_l + self.ctrl_entropy * self.loss_entropy_l
        # train high level
        if self.replay_buffer_h.start():
            self.loss_policy_h = 0.
            self.loss_value_h = 0.
            self.loss_goal_h = 0.
            high_train_steps = max(self.env.get_current_step() // self.k, 1)
            for _ in range(high_train_steps):
                loss_policy_h, loss_value_h, loss_goal_h = self.train_high_level()
                self.loss_policy_h += loss_policy_h
                self.loss_value_h += loss_value_h
                self.loss_goal_h += loss_goal_h
            self.loss_policy_h /= high_train_steps
            self.loss_value_h /= high_train_steps
            self.loss_goal_h /= high_train_steps
            self.loss_h = self.loss_policy_h + self.loss_value_h + self.goal_loss_coeff * self.loss_goal_h

        if self.loss_h is not None and self.loss_l is not None:
            self.loss = self.loss_h + self.loss_l

    def train_high_level(self):
        states, goals, rewards, next_states, dones = self.replay_buffer_h.sample()
        states = torch.from_numpy(states).float().to(self.device)
        goals = torch.from_numpy(goals).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).bool()

        next_goals = self.net.actor_h_tgt(next_states, explore_sigma=0.)
        next_vals_1, next_vals_2 = self.net.critic_h_tgt(next_states, next_goals)
        next_vals = torch.min(next_vals_1, next_vals_2)
        next_vals[dones] = 0.
        ref_vals = rewards + next_vals * self.man_discount
        vals_1, vals_2 = self.net.critic_h(states, goals)
        loss_value_h = F.mse_loss(vals_1, ref_vals.detach()) + F.mse_loss(vals_2, ref_vals.detach())

        self.optimizer_critic_h.zero_grad()
        loss_value_h.backward()
        self.optimizer_critic_h.step()

        curr_goals = self.net.actor_h(states, explore_sigma=0.)
        loss_policy_h = torch.tensor(0.).to(self.device)
        if self.policy_update_it % self.man_policy_update_freq == 0:
            loss_policy_h = -self.net.critic_h.value(states, curr_goals).mean()
        loss_goal_h = torch.clamp(F.pairwise_distance(self.a_net(states[:, :self.goal_dim]), self.a_net(curr_goals)) \
            - (self.r_margin_neg + self.r_margin_pos) / 2, min=0.).mean()
        loss_actor_h = loss_policy_h + self.goal_loss_coeff * loss_goal_h

        self.optimizer_actor_h.zero_grad()
        loss_actor_h.backward()
        self.optimizer_actor_h.step()

        self.net.soft_sync_high()
        self.policy_update_it += 1
        return loss_policy_h.item(), loss_value_h.item(), loss_goal_h.item()

    def train_low_level_a2c(self):
        states, goals, actions, info = self.memory_l.get_experience()
        next_states = states[1:]
        states = states[:-1]
        states = torch.cat(states, dim=0)
        next_states = torch.cat(next_states, dim=0)
        goals = torch.cat(goals, dim=0)
        rewards = self.compute_int_reward(states, goals, next_states)
        returns = self.compute_return(rewards, self.ctrl_discount, horizon=self.k)

        loss_policy_l = 0.
        loss_value_l = 0.
        loss_entropy_l = 0.
        for i in range(self.memory_l.size()):
            adv = returns[i] - info['value_l'][i]
            loss_policy_l -= adv.detach() * info['log_prob_act'][i]
            loss_value_l += adv.pow(2)
            entropy = -(info['prob'][i] * info['log_prob'][i]).sum(-1)
            loss_entropy_l -= entropy
        loss_policy_l /= self.memory_l.size()
        loss_value_l /= self.memory_l.size()
        loss_entropy_l /= self.memory_l.size()

        loss = loss_policy_l + loss_value_l + self.ctrl_entropy * loss_entropy_l
        self.optimizer_l.zero_grad()
        loss.backward()
        self.optimizer_l.step()
        return loss_policy_l.item(), loss_value_l.item(), loss_entropy_l.item()

    def compute_state_goal_similarity(self, states, next_states, goals):
        return -F.pairwise_distance(next_states[:, :self.goal_dim], goals)

    def compute_goal_reaching_reward(self, states, next_states, goals):
        diff = (next_states[:, :self.goal_dim] - goals).abs()
        return (diff <= 0.5 * torch.ones(self.goal_dim).to(self.device)).prod(dim=1).float()

    def compute_int_reward(self, states, goals, next_states):
        rewards = self.ctrl_rew_scale * self.compute_goal_reaching_reward(states, next_states, goals)
        self.mean_int_reward = float(rewards.mean())
        return rewards

    def compute_return(self, rewards, gamma=0.99, horizon=None):
        episode_length = len(rewards)
        returns = np.zeros(episode_length)
        if horizon is None:
            returns[episode_length - 1] = rewards[episode_length - 1]
            for i in reversed(range(episode_length - 1)):
                returns[i] = rewards[i] + returns[i + 1] * gamma
        else:
            for i in range(horizon, episode_length, horizon):
                returns[i-1] = rewards[i-1]
                for j in range(1, horizon):
                    returns[i-1-j] = rewards[i-1-j] + returns[i-j] * gamma
            returns[episode_length - 1] = rewards[episode_length - 1]
            j = 1
            while (episode_length - j) % horizon != 0:
                returns[episode_length - 1 - j] = rewards[episode_length - 1 - j] + returns[episode_length - j] * gamma
                j += 1
        return returns

    def test_one_episode(self):
        self.eval_env.new_episode()
        last_state = None
        last_goal = None
        goal_list = []
        last_obs_list = []
        obs_list = []
        done = False
        start_flag = self.replay_buffer_h.start()
        while True:
            flag = (self.eval_env.get_current_step() % self.k == 0)
            obs = self.eval_env.get_state()
            obs_var = utils.single_input_transform(obs, device=self.device)
            if flag:
                if self.eval_env.get_current_step() != 0:
                    goal_list.append(last_goal)
                    obs_list.append(obs_var)
                else:
                    last_obs_list.append(obs_var)

            action, last_goal, last_state, _ = self.step(
                obs_var, last_state, last_goal, flag, start_flag=start_flag, evaluate=True)
            
            self.eval_env.make_action(action)
            if self.eval_env.is_episode_finished():
                reward = self.eval_env.get_total_reward()
                break
        last_obs_list.extend(obs_list[:-1])
        goals = torch.cat(goal_list, dim=0)
        last_obs = torch.cat(last_obs_list, dim=0)
        obs = torch.cat(obs_list, dim=0)
        dist = -self.compute_state_goal_similarity(last_obs, obs, goals).mean().item()
        return reward, dist

    def update_adj_mat(self):
        for traj in self.traj_memory.get_trajectory():
            for i in range(len(traj)):
                for j in range(1, min(self.k, len(traj) - i)):
                    s1 = tuple(np.round(traj[i][:self.goal_dim]).astype(np.int32))
                    s2 = tuple(np.round(traj[i+j][:self.goal_dim]).astype(np.int32))
                    if s1 not in self.state_list:
                        self.state_list.append(s1)
                        self.state_dict[s1] = self.n_states
                        self.n_states += 1
                    if s2 not in self.state_list:
                        self.state_list.append(s2)
                        self.state_dict[s2] = self.n_states
                        self.n_states += 1
                    # assume that the environment is symmetric
                    self.adj_mat[self.state_dict[s1], self.state_dict[s2]] = 1
                    self.adj_mat[self.state_dict[s2], self.state_dict[s1]] = 1

    def test_adjacency_acc(self):
        states = torch.tensor(self.eval_env.states_all).float().to(self.device)
        self.a_net.eval()
        embeddings = self.a_net(states)
        dists = utils.euclidean_dist(embeddings, embeddings).detach().cpu().numpy()
        n_correct = 0
        n_total = 0
        r_dists = []
        for i in range(self.eval_env.n_states):
            r_dist = self.eval_env.calc_r_dist(self.eval_env.states_all[i])
            for j in range(i+1, self.eval_env.n_states):
                y = r_dist[tuple(self.eval_env.states_all[j])] <= self.k
                pred = dists[i, j] <= (self.r_margin_neg + self.r_margin_pos) / 2
                if pred == y:
                    n_correct += 1
                n_total += 1
        print('Adjacency acc = {:.4f}'.format(n_correct / n_total))
        self.a_net.train()

    def log_train(self):
        print("[@@ {} @@] ************** Training at episode {} **************".format(self.algo, self._episodes))
        utils.print_localtime()
        self.end_time = time.time()
        print('  Frames {}, Episode #{} (( costs {:.2f} s ))'.format(self._frames, self._episodes, self.end_time-self.start_time))
        if self.mean_int_reward is None:
            print('  Gets (( {:.3f} reward )) in (( {} steps ))'.format(self.curr_reward, self.env.get_current_step()))
        else:
            print('  Gets (( {:.3f} reward, {:.5f} mean int reward )) in (( {} steps ))'.format(
                self.curr_reward, self.mean_int_reward, self.env.get_current_step()))
        self.start_time = self.end_time
        if self.loss_h is not None:
            print("    High-level (( Policy loss = {:5f}  ||  Value loss = {:5f} ))".format(self.loss_policy_h, self.loss_value_h))
            print("               (( Goal loss = {:.5f} ))".format(self.loss_goal_h))
            self.summary_writer.add_scalar('high-level/policy loss', self.loss_policy_h, self._frames)
            self.summary_writer.add_scalar('high-level/value loss', self.loss_value_h, self._frames)
            self.summary_writer.add_scalar('high-level/goal loss', self.loss_goal_h, self._frames)
        else:
            print("    High-level (( Has not started training yet. Current replay size = {} ))".format(self.replay_buffer_h.size()))
        if self.loss_l is not None:
            print("    Low-level  (( Policy loss = {:5f}  ||  Value loss = {:5f} ))".format(self.loss_policy_l, self.loss_value_l))
            self.summary_writer.add_scalar('low-level/policy loss', self.loss_policy_l, self._frames)
            self.summary_writer.add_scalar('low-level/value loss', self.loss_value_l, self._frames)
            self.summary_writer.add_scalar('low-level/mean int reward', self.mean_int_reward, self._frames)
            print("               (( Entropy = {:5f} ))".format(-self.loss_entropy_l))
            self.summary_writer.add_scalar('low-level/entropy', -self.loss_entropy_l, self._frames)
        else:
            print("    Low-level  (( Has not started training yet. ))")

        if self.loss_h is not None and self.loss_l is not None:
            print("    Total loss = {:5f}".format(self.loss))
