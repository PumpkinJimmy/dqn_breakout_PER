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
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

        self.batch_size = batch_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_decay = beta_decay
        self.beta = self.beta_start
        
        self.heap = Heap(cmp=lambda a,b: a>b)
        self.pdf, self.ranges = self.build_dist()

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

        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)

        

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
        
        rnk = np.random.randint(self.ranges[:k]+1, self.ranges[1:])
        
        b_w = np.power(self.pdf[rnk-1] * self.__capacity, -self.beta)
        b_w = b_w.reshape(-1, 1).to(self.__device).float()
        
        indices = self.heap.get_eid_by_rnk(rnk)
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
    
        return b_state, b_action, b_reward, b_next, b_done, b_w
    
    # def calc_delta(self, state_batch, action_batch, next_batch, reward_batch, done_batch):
    #     values = self.__policy(state_batch.float()).gather(1, action_batch)
    #     values_next = self.__target(next_batch.float()).max(1).values.detach()
    #     expected = (self.__gamma * values_next.unsqueeze(1)) * \
    #         (1. - done_batch) + reward_batch

    def __len__(self) -> int:
        return self.__size
    
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
            ranges[s] = min(n, cur_r)
        return pdf, ranges
        # res = {}
        # n_partitions = self.batch_size
        # partition_num = 1
        # # each part size
        # partition_size = int(np.floor(self.__capacity / n_partitions))

        # for n in range(partition_size, self.size + 1, partition_size):
        #     distribution = {}
        #     # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        #     pdf = np.pow(np.arnage(1, n+1), -self.alpha)
            
        #     pdf_sum = np.sum(pdf)
        #     distribution['pdf'] = pdf / pdf_sum
        #     # split to k segment, and than uniform sample in each k
        #     # set k = batch_size, each segment has total probability is 1 / batch_size
        #     # strata_ends keep each segment start pos and end pos
        #     cdf = np.cumsum(distribution['pdf'])
        #     strata_ends = np.zeros((self.batch_size+1, ))
        #     strata_ends[self.batch_size] = self.__size
        #     step = 1 / float(self.batch_size)
        #     index = 1
        #     for s in range(2, self.batch_size + 1):
        #         while cdf[index] < step:
        #             index += 1
        #         strata_ends[s] = index
        #         step += 1 / float(self.batch_size)

        #     distribution['strata_ends'] = strata_ends

        #     res[partition_num] = distribution

        #     partition_num += 1

        # return res
        