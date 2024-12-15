#specify

import numpy as np
import torch
import torch.nn as nn

from flow.flow_code import made  #kamen flows
import warnings

class Flow(nn.Module):
    def __init__(self, flow_type=None, flow_args=None, flow_norm=None, device='cpu', expert_replay=None, env_name=None): #the nones are there so the loading works...
        super().__init__()
        self.flow_type = flow_type  # flow type must clearly identiy what model from which source...
        self.flow_args = flow_args
        self.flow_norm = flow_norm  # none, sch, vd, ... (for now, not handling sch, due to random...
        self.device = device
        self.env_name = env_name
        
        #create flow and send to the device
        if self.flow_type in ['MAF','MAFMOG','MADE','MADEMOG','RealNVP']:
            self.flow = getattr(made, self.flow_type)(**self.flow_args).to(device)
        else:
            raise ValueError("only kamen flows are currently supported here")

        if flow_norm == 'vd':
            if expert_replay == None:
                warnings.warn('flow_norm is vd and you didnt set the scale and shift params...')
                self.scale = None
                self.shift = None

        if flow_norm == 'sch':
            self.input_min = None
            self.input_max = None

        self.to(device)

        
    def log_prob(self, data):
        #normalize
        data = self.normalize(data)
        return self.flow.log_prob(data)

    def sample(self, data):
        #normalize
        data = self.normalize(data)
        return self.flow.sample(data)
    
    #(remember, for now we don't use generative side, but in future might need to reverse... (also the normalization transforms the density and we arent really taking that into account (but i guess if q function and actor use it than its fine) (beware of an RL that already normalizes, this will be problematic for our use...)
    
    def normalize(self, data):
        if self.flow_norm == "none":
            return data
        elif self.flow_norm == "vd":
            return (data + self.shift) * self.scale
        elif self.flow_norm == "sch":
            return (data - self.input_min)/(self.input_max - self.input_min) * 2 - 1
    
    
    #add saving and loading...
    #perhaps only needed in the wrapping class...
    #note how double flow will also need that...
