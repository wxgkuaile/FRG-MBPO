import torch
import torch.nn as nn


class Rewarder(nn.Module):
    def __init__(self, model, update_every=1000, update_iters=10, update_batch_size=100, debug=False):
        super().__init__()
        self.model = model
        # self.env_pool = env_pool
        # self.model_pool = model_pool
        self.option = model.option
        self.device = next(self.model.parameters()).device
        # self.device = device

        self.update_every = update_every
        self.update_iters = update_iters
        self.update_batch_size = update_batch_size

        self.debug = debug

    def get_reward(self, batch, not_rl=False):
        if not not_rl:
            state = torch.Tensor(batch[0]).to(self.device)
            action = torch.Tensor(batch[1]).to(self.device)
            reward = torch.Tensor(batch[2]).unsqueeze(1).to(self.device)

            if self.option == 1:
                batch = torch.cat((state, action, reward), dim=1)
            elif self.option == 0:
                batch = torch.cat((state, reward), dim=1)

        r = self.model.get_reward(batch)

        return r

    # only flow predict
    def get_next_state(self, batch, not_rl=False):
        if not not_rl:
            state = torch.Tensor(batch[0]).to(self.device)
            action = torch.Tensor(batch[1]).to(self.device)
            reward = torch.Tensor(batch[2]).unsqueeze(1).to(self.device)

            if self.option == 1:
                batch = torch.cat((state, action), dim=1)
            elif self.option == 0:
                batch = torch.cat((state, reward), dim=1)

        next_state = self.model.get_next_state(batch)

        return next_state

    def update(self, env_pool, model_pool):

        # print('updating flow')
        self.model.update(env_pool, model_pool, device=self.device, iterations=self.update_iters,
                          batch_size=self.update_batch_size)
        # print('done updating flow')
