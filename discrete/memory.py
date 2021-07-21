import numpy as np


class HighLevelReplayBuffer:

    def __init__(self, capacity, batch_size, obs_dim, action_dim):
        self._capacity = capacity
        self._batch_size = batch_size
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.zeros((self._capacity, self._obs_dim))
        self.next_state = np.zeros((self._capacity, self._obs_dim))
        self.action = np.zeros((self._capacity, self._action_dim))
        self.reward = np.zeros(self._capacity, dtype=np.float32)
        self.done = np.zeros(self._capacity, dtype=np.uint8)

        self._ptr = 0
        self._size = 0

    def append(self, state, action, reward, next_state, done):
        self.state[self._ptr] = state
        self.next_state[self._ptr] = next_state
        self.action[self._ptr] = action
        self.reward[self._ptr] = reward
        self.done[self._ptr] = done

        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self):
        ind = np.random.choice(self._size, self._batch_size, replace=False)
        return self.state[ind], self.action[ind], self.reward[ind], self.next_state[ind], self.done[ind]

    def size(self):
        return self._size

    def start(self):
        return self._size >= 0.5 * self._capacity

    def full(self):
        return self._size >= self._capacity


class LowLevelMemory:

    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.goals = []
        self.actions = []
        self.info = dict(prob=[], log_prob=[], log_prob_act=[], value_l=[])

        self._size = 0

    def append(self, state, goal, action, info):
        self.states.append(state)
        self.goals.append(goal)
        self.actions.append(action)
        self.info['prob'].append(info['prob'])
        self.info['log_prob'].append(info['log_prob'])
        self.info['log_prob_act'].append(info['log_prob_act'])
        self.info['value_l'].append(info['value_l'])

        self._size += 1

    def get_experience(self):
        return self.states, self.goals, self.actions, self.info

    def size(self):
        return self._size


class TrajectoryMemory:

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
