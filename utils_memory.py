from typing import (
    Any,
    Tuple,
)
import numpy as np
import torch
from heap import Heap

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)


class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        indices = torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size



class PrioritizedReplayMemory:
    '''
    WARNING: 必须要有满了才允许开始采样
    '''
    def __init__(self, 
    channels: int, 
    capacity: int, 
    device: TorchDevice,
    batch_size = 32,
    alpha: float = 0.7,
    beta_start: float = 0.5,
    beta_decay: int = 100_000) -> None:
        self.__device = device
        # max len
        self.__capacity = capacity
        # current len
        self.__size = 0
        self.__pos = 0

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8,
            device=self.__device)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long,
        device=self.__device)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8,
        device=self.__device)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool,
        device=self.__device)

        self.batch_size = batch_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_decay = beta_decay
        self.beta = self.beta_start
        
        self.heap = Heap()
        self.pdf, self.ranges = self.build_dist()
        self.pdf = torch.Tensor(self.pdf).to(self.__device)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
            delta: float
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        self.heap.update(np.abs(delta), self.__pos)

        self.__size = max(self.__size, self.__pos+1)
        self.__pos = (self.__pos + 1) % self.__capacity
        

        

    def sample(self) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
            Any
    ]:
        batch_size = self.batch_size
        # indices = torch.randint(0, high=self.__size, size=(batch_size,))
        self.beta = min(self.beta + (1.0 - self.beta_start)/self.beta_decay, 1.0)
        k = self.batch_size
        
        rnk = np.random.randint(self.ranges[:k]+1, self.ranges[1:]+1)
        
        b_w = torch.pow(self.pdf[rnk-1] * self.__capacity, -self.beta)
        b_w = b_w.reshape(-1, 1).float()
        b_w /= b_w.max()
        
        indices = self.heap.get_eid_by_rnk(rnk)
        b_state = self.__m_states[indices, :4].float()
        b_next = self.__m_states[indices, 1:].float()
        b_action = self.__m_actions[indices]
        b_reward = self.__m_rewards[indices].float()
        b_done = self.__m_dones[indices].float()
    
        return b_state, b_action, b_reward, b_next, b_done, b_w
    

    def __len__(self) -> int:
        return self.__size
    
    def full(self):
        return self.__size == self.__capacity
    
    def sort(self):
        self.heap.sort()
    
    def build_dist(self):
        """
        preprocess pow of rank
        (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        分k段
        只产生最大容量时的预处理分布
        假设最大容量远大于batch大小
        同一个排名可能被选中多次
        :return: distributions, dict
        """
        k = self.batch_size
        n = self.__capacity
        # pdf[i] = PDF(i-1)
        pdf = np.power(np.arange(1, n+1), -self.alpha)
        pdf /= pdf.sum()
        # cdf[i] = CDF(i-1)
        cdf = pdf.cumsum()
        # range(i) = [ranges[i]+1, ranges[i+1]]
        # ls = ranges[:k]+1
        # rs = ranges[1:]
        ranges = np.zeros((k+1, ))
        ranges[k] = n
        step = 1/ float(self.batch_size)
        cur_r = 1
        # for s in [1, k-1]
        for s in range(1, k):
            while cur_r-1 < cdf.shape[0] and cdf[cur_r-1] < s*step:
                cur_r += 1
            ranges[s] = min(n-1, cur_r)
        return pdf, ranges
        