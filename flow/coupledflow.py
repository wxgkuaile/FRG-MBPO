import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math


class CoupledFlow(nn.Module):
    def __init__(self, flow1, flow2, lr, option=1, device='cpu', use_tanh=False, tanh_scale=(1, 1),
                 tanh_shift=False, flow_reg=False, flow_reg_weight=1, smooth=None, env_name=None,
                 rewarder_replay_size=None):
        super(CoupledFlow, self).__init__()

        self.option = option

        self.env_name = env_name
        self.rewarder_replay_size = rewarder_replay_size

        self.use_tanh = use_tanh
        self.tanh_scale = tanh_scale[0]
        self.tanh_unscale = tanh_scale[1]
        self.tanh_shift = tanh_shift
        self.flow_reg = flow_reg
        self.flow_reg_weight = flow_reg_weight
        self.smooth = smooth

        self.flow1 = flow1
        self.flow2 = flow2

        self.optimizer = torch.optim.Adam(list(self.flow1.parameters()) + list(self.flow2.parameters()),
                                          lr=lr)
        self.model_parameters = list(self.flow1.parameters()) + list(self.flow2.parameters())

        self.to(device)

    def get_flow1_log_probs(self, data):
        if self.env_name == 'Ant-v2':  # -----------------SPECIAL  HANDLING OF ANT DUE TO NO CONTACT FORCES https://www.gymlibrary.ml/environments/mujoco/ant/
            if self.option == 1:
                data = torch.cat((data[:, :27], data[:, -8:]), dim=1)  # first 27 dims and last 8 actions
            elif self.option == 0:
                data = data[:, :27]  # first 27 dims of obs
            elif self.option == 2:
                data = torch.cat((data[:, :27], data[:, 111:138]), dim=1)  # first 27 of obs and 27 of next obs
        elif self.env_name == 'Humanoid-v2':
            if self.option == 1:
                b = np.array([True] * 17)  # last 17 action dims
                temp = np.concatenate((self.idx, b))
                data = data[:, temp-1]
            elif self.option == 0:
                data = data[:, self.idx]
            elif self.option == 2:
                temp = np.concatenate((self.idx, self.idx))
                data = data[:, temp]
        else:
            return self.flow1.log_prob(data)
        return self.flow1.log_prob(data)

    def get_flow2_log_probs(self, data):
        if self.env_name == 'Ant-v2':  # -----------------SPECIAL HANDLING OF ANT DUE TO NO CONTACT FORCES https://www.gymlibrary.ml/environments/mujoco/ant/
            if self.option == 1:
                data = torch.cat((data[:, :27], data[:, -8:]), dim=1)  # first 27 dims and last 8 actions
            elif self.option == 0:
                data = data[:, :27]  # first 27 dims of obs
            elif self.option == 2:
                data = torch.cat((data[:, :27], data[:, 111:138]), dim=1)  # first 27 of obs and 27 of next obs
        elif self.env_name == 'Humanoid-v2':
            if self.option == 1:
                b = np.array([True] * 17)  # last 17 action dims
                temp = np.concatenate((self.idx, b))
                data = data[:, temp-1]
            elif self.option == 0:
                data = data[:, self.idx]
            elif self.option == 2:
                temp = np.concatenate((self.idx, self.idx))
                data = data[:, temp]
        else:
            return self.flow2.log_prob(data)

        return self.flow2.log_prob(data)

    def get_next_state(self, data):
        return self.flow1.sample(data)

    def x(self, data, training=False):
        a = self.get_flow1_log_probs(data)
        b = self.get_flow2_log_probs(data)
        x = a - b

        if self.use_tanh:
            return self.tanh_unscale * F.tanh(x / self.tanh_scale)

        return x

    def calc_loss(self, p, q):  # p is env data, q is model data  #213
        # a = self.x(p).exp().mean().log()
        a = torch.logsumexp(self.x(p, training=True), dim=0) - math.log(p.shape[0])
        b = self.x(q, training=True).mean()
        loss = a - b

        return loss  # see value dice equation 7

    def get_reward(self, batch):
        # -x is the reward
        with torch.no_grad():
            r = -self.x(batch)

        if self.tanh_shift:
            return r + self.tanh_unscale

        #r = F.normalize(r, dim=0)

        return r

    def smoother(self, data):
        if self.smooth:
            return data + (self.smooth) * ((data + 0.001) * (torch.rand(data.shape).to(
                data.device) - 1 / 2))  # smooth each dimension of the state with uniform noise scaled to its value
        else:
            return data



    def update(self, env_pool, model_pool, device, iterations=10, batch_size=100, not_rl=False):
        # data1 - env_data. data2 - model_data.
        for t in range(iterations):

            batch1 = env_pool.sample(batch_size=batch_size, rewarder=None)
            batch2 = model_pool.sample_all_batch(batch_size=batch_size)

            env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(batch_size=batch_size, rewarder=None)
            env_state = torch.Tensor(env_state).to(device)
            env_action = torch.Tensor(env_action).to(device)
            env_reward = torch.Tensor(env_reward).unsqueeze(1).to(device)
            #env_next_state = torch.Tensor(env_next_state).to(device)

            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(batch_size=batch_size)
            model_state = torch.Tensor(model_state).to(device)
            model_action = torch.Tensor(model_action).to(device)
            model_reward = torch.Tensor(model_reward).to(device)
            #model_next_state = torch.Tensor(model_next_state).to(device)

            if self.option == 1:

                batch1 = torch.cat((env_state, env_action, env_reward), dim=1)
                batch2 = torch.cat((model_state, model_action, model_reward), dim=1)

            elif self.option == 0:
                batch1 = torch.cat((env_state, env_reward), dim=1)
                batch2 = torch.cat((model_state, model_reward), dim=1)

            batch1 = self.smoother(batch1)
            batch2 = self.smoother(batch2)

            loss = self.calc_loss(batch1, batch2)

            if self.flow_reg:
                losses = self.get_flow1_log_probs(batch2.to(device))
                loss11 = -losses.mean()

                losses = self.get_flow2_log_probs(batch1.to(device))
                loss22 = -losses.mean()

                loss = loss + (loss11 + loss22) * self.flow_reg_weight

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_parameters, max_norm=1)
            self.optimizer.step()
