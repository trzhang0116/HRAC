import numpy as np
from collections import deque
from gym import spaces


class BaseEnv:

    def __init__(self, random_start=False, seed=0):
        self.random_start = random_start
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self.h = None
        self.w = None
        self.world = None
        self.states_all = None
        self.n_states = None
        self.start_pos = None
        self.start_states = None

    def reset(self):
        if self.random_start:
            self.state = self.start_states[self.rng.randint(self.n_states-1)]
        else:
            self.state = self.start_pos
        self.r_total = 0.
        self.done = False
        self.info = None
        self._step = 0
        self.last_action = None
        return self.state
  
    def step(self, action):
        pass

    def new_episode(self):
        self.reset()

    def get_state(self):
        return self.state

    def make_action(self, action):
        _, r, _, _ = self.step(action)
        return r

    def is_episode_finished(self):
        return self.done

    def get_total_reward(self):
        return self.r_total

    def get_current_step(self):
        return self._step

    def render(self):
        raise NotImplementedError

    def calc_r_dist(self, target):
        r_dist = np.Inf * np.ones((self.h, self.w)).astype(np.uint8)
        flags = np.zeros((self.h, self.w), dtype=bool)
        r_dist[target[0], target[1]] = 0
        buf = deque()
        for p in self._adj_pos(target):
            if r_dist[p[0], p[1]] == np.Inf:
                buf.append(p)
                flags[p[0], p[1]] = True
        while buf:
            pos = buf.popleft()
            dists = [r_dist[p[0], p[1]] + 1 for p in self._adj_pos(pos)]
            r_dist[pos[0], pos[1]] = np.min(dists)
            for p in self._adj_pos(pos):
                if not flags[p[0], p[1]] and r_dist[p[0], p[1]] == np.Inf:
                    buf.append(p)
                    flags[p[0], p[1]] = True
        return r_dist

    def get_adj_mat(self, k):
        adj_mat = np.zeros((self.n_states, self.n_states))
        for i, s in enumerate(self.states_all):
            r_dist = self.calc_r_dist(s)
            for j, t in enumerate(self.states_all):
                if r_dist[t[0], t[1]] <= k:
                    adj_mat[i, j] = 1
        return adj_mat

    def _adj_pos(self, pos):
        pos_list = []
        for i in range(4):
            next_pos = pos + self.acts[i]
            if self.world[next_pos[0], next_pos[1]] != 1:
                pos_list.append(next_pos)
        return pos_list


class MazeEnv(BaseEnv):

    def __init__(self, step_limit=200, reward_shaping=True, reward_shaping_scale=0.1,
                 random_start=False, pure_exploration=False, random_action_prob=0.25, seed=0):
        super().__init__(random_start=random_start, seed=seed)
        layout = """\
wwwwwwwwwwwwwwwww
w               w
w               w
w  wwwwwwwwwww  w
w  wG        w  w
w  w         w  w
w  wwwwwwww  w  w
w            w  w
w            w  w
wwwwwwwwwwwwww  w
w               w
wS              w
wwwwwwwwwwwwwwwww
"""
        self.step_limit = step_limit
        self.reward_shaping = reward_shaping
        self.reward_shaping_scale = reward_shaping_scale
        self.pure_exploration = pure_exploration
        self.random_action_prob = random_action_prob

        self.obs_dim = 2
        self.action_dim = 4

        self.world = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])
        self.h, self.w = self.world.shape
        for i, line in enumerate(layout.splitlines()):
            for j, c in enumerate(line):
                if c == 'S':
                    self.world[i, j] = 2
                    self.start_pos = np.array((i, j))
                elif c == 'G':
                    self.world[i, j] = 3
                    self.goal_pos = np.array((i, j))

        self.acts = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=max(self.h, self.w)-1, shape=(self.obs_dim, ), dtype=np.uint8)        
                 
        self.n_states = 0
        self.start_states = []
        self.states_all = []
        self.states_walls_all = []
        self.state_inds_dict = {}
        self.state_wall_inds_dict = {} 
        for i in range(self.h):
            for j in range(self.w):
                self.states_walls_all.append(np.array((i, j)))
                self.state_wall_inds_dict[(i, j)] = self.n_states
                if self.world[i, j] != 1:
                    self.states_all.append(np.array((i, j)))
                    self.state_inds_dict[(i, j)] = self.n_states
                    if self.world[i, j] in [0, 2]:
                        self.start_states.append(np.array((i, j)))
                    self.n_states += 1

        if self.reward_shaping:
            self.r_dist = self.calc_r_dist(self.goal_pos)
  
    def step(self, action):
        if not self.done:
            r = 0.
            state = self.state
            if self.rng.rand() < self.random_action_prob:
                action = self.rng.randint(self.action_dim)
            self.last_action = action
            next_state = self.state + self.acts[action]
            if self.world[next_state[0], next_state[1]] != 1:  # not wall
                self.state = next_state
                if self.world[next_state[0], next_state[1]] == 3:  # goal reached
                    r = 0.1
                    if not self.pure_exploration:
                        self.done = True
                elif self.reward_shaping:
                    r = self.reward_shaping_scale * self._relative_dist(state, next_state)
                self.r_total += r

            self._step += 1
            if self._step >= self.step_limit:
                self.done = True
            return self.state, r, self.done, self.info

    def _relative_dist(self, last_pos, pos):
        last_dist = self.r_dist[last_pos[0], last_pos[1]]
        dist = self.r_dist[pos[0], pos[1]]
        if last_dist > dist:
            return 1
        if last_dist < dist:
            return -1
        return 0


class KeyChestEnv(BaseEnv):

    def __init__(self, step_limit=500, random_start=False, pure_exploration=False,
                 random_action_prob=0.25, seed=0):
        super().__init__(random_start=random_start, seed=seed)
        layout = """\
wwwwwwwwwwwwwwwww
w               w
wB              w
wwwwwwwwwwwwww  w
w               w
w               w
wK wwwwwwwwwww  w
w               w
w               w
wwwwwwwwwwwwww  w
w               w
w      S        w
wwwwwwwwwwwwwwwww
"""
        self.step_limit = step_limit
        self.pure_exploration = pure_exploration
        self.random_action_prob = random_action_prob

        self.obs_dim = 3
        self.action_dim = 4

        self.world = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])
        self.h, self.w = self.world.shape

        for i, line in enumerate(layout.splitlines()):
            for j, c in enumerate(line):
                if c == 'S':
                    self.world[i, j] = 2
                    self.start_pos = np.array((i, j))
                elif c == 'G':
                    self.world[i, j] = 3
                    self.goal_pos = np.array((i, j))
                elif c == 'K':
                    self.world[i, j] = 4
                    self.key_pos = np.array((i, j))
                elif c == 'B':
                    self.world[i, j] = 5
                    self.box_pos = np.array((i, j))

        self.acts = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=max(self.h, self.w)-1, shape=(self.obs_dim, ), dtype=np.uint8)        
                 
        self.n_states = 0
        self.start_states = []
        self.states_all = []
        self.states_walls_all = []
        self.state_inds_dict = {}
        self.state_wall_inds_dict = {} 
        for i in range(self.h):
            for j in range(self.w):
                self.states_walls_all.append(np.array((i, j)))
                self.state_wall_inds_dict[(i, j)] = self.n_states
                if self.world[i, j] != 1:
                    self.states_all.append(np.array((i, j)))
                    self.state_inds_dict[(i, j)] = self.n_states
                    self.start_states.append(np.array((i, j)))
                    self.n_states += 1

    def reset(self):
        if self.random_start:
            self.state = self.start_states[self.rng.randint(self.n_states-1)]
        else:
            self.state = self.start_pos
        self.r_total = 0.
        self.done = False
        self.info = None
        self._step = 0
        self.last_action = None
        self.has_key = 0
        return np.append(self.state, self.has_key)
  
    def step(self, action):
        if not self.done:
            r = 0.
            state = self.state
            if self.rng.rand() < self.random_action_prob:
                action = self.rng.randint(self.action_dim)
            self.last_action = action
            next_state = self.state + self.acts[action]
            if self.world[next_state[0], next_state[1]] != 1:  # not wall
                self.state = next_state
                if self.world[next_state[0], next_state[1]] == 4:  # pick up the key
                    if self.has_key == 0:
                        r = 1.
                        self.has_key = 1
                if self.world[next_state[0], next_state[1]] == 5 and self.has_key == 1:  # reach the box with the key
                    r = 5.
                    if not self.pure_exploration:
                        self.done = True
                self.r_total += r

            self._step += 1
            if self._step >= self.step_limit:
                self.done = True
            return np.append(self.state, self.has_key), r, self.done, self.info

    def get_state(self):
        return np.append(self.state, self.has_key)
