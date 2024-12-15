import random
import numpy as np
from operator import itemgetter

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def clear(self):
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * int(append_len))

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position

    def sample(self, batch_size, rewarder=None):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        if rewarder:
           batch = [state, action, reward, next_state, done]
           reward_flow = rewarder.get_reward(batch).detach().cpu()
           #reward = (1- 1/batch_size)*reward + (1/batch_size)*reward_flow #
           reward = reward + reward_flow # + reward_flow
           print("reward:", reward.mean(0))
        return state, action, reward, next_state, done

    def sample_all_batch(self, batch_size, rewarder=None):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        if rewarder:
          batch = [state, action, reward, next_state, done]
          reward = rewarder.get_reward(batch).detach().cpu()

        return state, action, reward, next_state, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)
