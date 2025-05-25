
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
import time

class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args
        # input_shape = 24 = 观测6+上一次动作13+智能体独热编码5

        self.obs_dim = 6
        self.action_dim = 13
        self.id_dim = 5
        
        self.dragon_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
        )
        self.self_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(16+16+32+5, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, args.n_actions)
        )
        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.dragon_encoder)
            orthogonal_init_(self.self_encoder)
            orthogonal_init_(self.action_encoder)
            orthogonal_init_(self.fc1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # b, a, e = inputs.size()

        # inputs = inputs.view(-1, e)
        # x = F.relu(self.fc1(inputs), inplace=True)
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # hh = self.rnn(x, h_in)

        # if getattr(self.args, "use_layer_norm", False):
        #     q = self.fc2(self.layer_norm(hh))
        # else:
        #     q = self.fc2(hh)

        # return q.view(b, a, -1), hh.view(b, a, -1)
        b, a, e = inputs.shape
        
        inputs = inputs.view(-1, e)
        obs = inputs[..., :self.obs_dim]
        actions = inputs[..., self.obs_dim:self.obs_dim+self.action_dim]
        agent_ids = inputs[..., -self.id_dim:]
        
        dragon_feat = self.dragon_encoder(obs[..., 3:6].view(-1,3))
        self_feat = self.self_encoder(obs[..., :3].view(-1,3))
        actions_feat = self.action_encoder(actions.view(-1,13))
        combined = th.cat([
            dragon_feat, 
            self_feat,
            actions_feat,
            agent_ids
        ], dim=-1)
        combined = self.fc1(combined)
        
        hh = self.rnn(combined, hidden_state.view(-1, self.args.rnn_hidden_dim))
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)

        q = self.fc2(hh)
        
        return q.view(b, a, -1), hh.view(b, a, -1)
