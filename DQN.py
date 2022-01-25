import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json

class FCN(nn.Module):
    """
    fully connected network for Q-network.
    """
    def __init__(self, state_dim, action_dim, dueling=False):
        """
        init a Q-network.

        if `dueling=True`, it will be a Q-network of Dueling DQN.
        """
        super(FCN, self).__init__()
        self.dueling = dueling
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )

        if self.dueling:
            self.advantage = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            self.value = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )

        else:
            self.value = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )

    def forward(self, x):
        out = self.feature(x)

        if self.dueling:
            v = self.value(out)
            adv = self.advantage(out)
            adv_mean = torch.mean(adv, dim=1, keepdim=True)
            out = adv + v - adv_mean
        
        else:
            out = self.value(out)

        return out

class ISWeightMSELoss(nn.Module):
    """
    MSELoss with ISWeight.
    """

    def __init__(self):
        super(ISWeightMSELoss, self).__init__()

    def forward(self, x, y, w):
        """
        mean(w * (x - y)^2)
        """

        loss = torch.mean(w * torch.pow((x - y), 2))
        return loss

class ReplayBuffer:
    """
    Normal Experience Replay Buffer.
    """

    def __init__(self, capacity, data_size):
        """
        init an Experience Replay Buffer
        """
        self.capacity = capacity
        self.data_size = data_size
        self.memory = np.empty((self.capacity, self.data_size))
        self.length = 0
        self.index = 0

    def push(self, transition):
        """
        push a transition into memory. 
        
        if it is filled, the old transition will be overrode.
        """
        if self.length < self.capacity:
            self.length += 1
        else:
            self.index = 0
        self.buffer[self.index] = transition
        self.index += 1

    def sample(self, batch_size):
        """
        sample a batch of transition
        """
        batch = np.array(random.sample(self.buffer, batch_size))
        return batch

class SumTree:
    """
    SumTree of Prioritized Replay, to store p and transition.
    """

    def __init__(self, size, data_size):
        """
        size: capacity of memory.

        data_size: dimension of transition.
        """

        y = len(str(bin(size))) - 2
        self.index = 0
        self.size = 2 ** y    # it should be a full binary tree.
        self.tree = np.zeros(self.size * 2 - 1)
        self.data = np.empty((size, data_size))
        self.update(0, 1)    # p1 = 1

    def append(self, data, p):
        """
        add data and it's p to SumTree.
        """
        if self.index >= self.size:
            self.index = 0
        self.data[self.index] = data
        self.update(self.index, p)
        self.index += 1

    def update(self, index, p):
        """
        update all nodes of tree, with transition's p and it's index in data memory.
        """

        index = self.size - 1 + index    # to index of tree.
        delta = p - self.tree[index]
        self.tree[index] = p
        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += delta

    def get_leaf(self, value):
        """
        get the leaf node in interval by the value which is sampled.

        return:
        1. index: index of transition in data memory
        2. data: data of transition
        3. p: p of transition
        """

        index = 0    # from top
        while index < (self.size - 1):    # last layer: [n-1] ·········· [2n-1]
            left = index  * 2 + 1
            right = left + 1
            # left child
            if value <= self.tree[left]:
                index = left 
            # right child   
            else:
                value -= self.tree[left]
                index = right    
        
        p = self.tree[index]
        index = index - self.size + 1
        data = self.data[index]
        return index, data, p

    def get_sum_p(self):
        """
        sum p of the tree.
        """
        return self.tree[0]

    def get_max_p(self):
        """
        max p of last layer.
        """
        return np.max(self.tree[-self.size:])

    def get_min_p(self):
        """
        min p of last layer.
        """
        min_p = np.min(self.tree[-self.size:])
        return min_p if min_p != 0 else 1

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    """

    def __init__(self, capacity, data_size, alpha=None, beta=0.4, epsilon=0.01):
        """
        init a Prioritized Experience Replay Buffer.
        """
        self.capacity = capacity
        self.data_size = data_size
        self.length = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.memory = SumTree(self.capacity, self.data_size)

    def push(self, transition):
        """
        push a transition into memory. 
        
        if it is filled, the old transition will be overrode.
        """
        if self.length < self.capacity:
            self.length += 1
        p = self.memory.get_max_p()
        self.memory.append(transition, p)

    def sample(self, batch_size):
        """
        sample a batch of transition

        return:
            1. index_batch,
            2. transition_batch,
            3. ISWeight_batch
        """
        index_batch = np.empty((batch_size, 1), dtype=np.int32)
        transition_batch = np.empty((batch_size, self.data_size))
        ISWeight_batch = np.empty((batch_size, 1))

        sum_p = self.memory.get_sum_p()
        min_p = self.memory.get_min_p()
        min_prob = min_p / sum_p    # min P(i)
        interval = sum_p / batch_size
        for n in range(batch_size):
            low = interval * n
            high = interval * (n + 1)
            rand = np.random.uniform(low, high)    # a random number in [low, high)
            index, transition, p = self.memory.get_leaf(rand)
            prob = p / sum_p    # probability P(j)
            ISWeight = np.power(prob/min_prob, -self.beta)    # weight = (P(j) / min P(i)) ^ (-beta)

            index_batch[n] = index
            transition_batch[n] = transition
            ISWeight_batch[n] = ISWeight
        return index_batch, transition_batch, ISWeight_batch

    def update(self, index_list, error_list):
        """
        update SumTree of buffer.
        """
        length = len(index_list)
        for i in range(length):
            index = index_list[i]
            p = abs(error_list[i]) + self.epsilon
            self.memory.update(index, p)

class Agent:
    """
    agent with DQN moudle.
    """
    def __init__(self, state_dim, action_dim, cuda=False, dueling=False, prioritized=False):
        """
        init an agent with DQN moudle.

        - if `cuda=True`, the GPU will be used if it is available.
        - if `dueling=True`, it will be with Dueling DQN moudle.
        - if `prioritized=True`, the replay buffer will be a Prioritized Replay Buffer.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_dueling = dueling
        self.is_prioritized = prioritized
        self.train_cfg = {    # configuration of training.
            "rate" : 0.001, 
            "gamma" : 0.9, 
            "epsilon" : 0.9, 
            "capacity" : 100000,
            "batch_size" : 128, 
            "target_replace_iter" : 200
        }
        if torch.cuda.is_available() and cuda == True:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.predict_net = FCN(self.state_dim, action_dim, dueling=dueling).to(self.device)
        self.target_net = FCN(self.state_dim, action_dim, dueling=dueling).to(self.device)
        self.transition_size = self.state_dim * 2 + 3
        if prioritized:
            self.memory = PrioritizedReplayBuffer(self.train_cfg["capacity"], self.transition_size)
            self.loss = ISWeightMSELoss()
        else:
            self.memory = ReplayBuffer(self.train_cfg["capacity"], self.transition_size)
            self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.predict_net.parameters(), lr=self.train_cfg["rate"])
            
        self.learn_step_counter = 0    # to update Q-netwark

    def choose_action(self, state, greedy=True):
        """
        choose an action depend on state.

        if `greedy = True`, epsilon greedy will be used.

        return a integer from [0, action_dim]
        """
        if np.random.rand() > (1 - self.train_cfg["epsilon"]) and greedy:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float32, device=self.device)    # shape(1, state_dim)
            q = self.predict_net(state)
            action = q.max(1)[1].item()     # index
        return action

    def update(self):
        """
        update moudle for learning.
        """
        if self.memory.length < self.train_cfg["batch_size"]*2:
            return

        if self.learn_step_counter % self.train_cfg["target_replace_iter"] == 0:
            self.target_net.load_state_dict(self.predict_net.state_dict())
        self.learn_step_counter += 1

        if self.is_prioritized:
            index_batch, data_batch, ISWeight_batch = self.memory.sample(self.train_cfg["batch_size"])
            ISWeight_batch = torch.tensor(ISWeight_batch, dtype=torch.float32, device=self.device)
        else:
            data_batch = self.memory.sample(self.train_cfg["batch_size"])
        
        state_batch = torch.tensor(data_batch[:, :self.state_dim], dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(data_batch[:, self.state_dim], dtype=torch.int64, device=self.device).unsqueeze(1)  # shape: (n, 1)
        reward_batch = torch.tensor(data_batch[:, self.state_dim+1], dtype=torch.float32, device=self.device) #  shape: (n)
        next_state_batch = torch.tensor(data_batch[:, -self.state_dim-1:-1], dtype=torch.float32, device=self.device)
        terminated_batch = torch.tensor(data_batch[:, -1], dtype=torch.int64, device=self.device)

        q = self.predict_net(state_batch).gather(1, action_batch)
        next_q = self.target_net(next_state_batch).detach()
        q_ = (reward_batch + self.train_cfg["gamma"] * next_q.max(1)[0] * (1 - terminated_batch)).unsqueeze(1)

        if self.is_prioritized:
            loss = self.loss(q, q_, ISWeight_batch)    # 1/m sum(wj * (q - q_)^2)
            td_error = (q - q_).detach().cpu()    # copy data from GPU to CPU
            td_error = td_error.numpy().T[0]    # to 1-dim array
            self.memory.update(index_batch, td_error)
        else:
            loss = self.loss(q, q_)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data.item()

    def store(self, state, action, reward, next_state, terminated):
        """
        store a transiton with (s, a, r, s_, done)
        """
        transition = np.hstack((state, action, reward, next_state, terminated))
        self.memory.push(transition)

    def save(self, path):
        """
        save network to a file.
        """
        torch.save(self.predict_net.state_dict(), path)

    def load(self, path):
        """
        load network from a file.
        """
        self.predict_net.load_state_dict(torch.load(path))

    def read_config(self, path):
        """
        load configurations from a JSON file for trainning.
        """
        with open(path) as f:
            config_load = json.loads(f.read())
        for key in config_load.keys():
            if key in self.train_cfg.keys():
                self.train_cfg[key] = config_load[key]
