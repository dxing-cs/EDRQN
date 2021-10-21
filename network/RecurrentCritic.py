import torch
import torch.nn as nn


class RecurrentCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(RecurrentCritic, self).__init__()
        self.input_dim = sum(input_dim) if isinstance(input_dim, list) else input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        self.init_func = nn.init.orthogonal_

        self.linear_1 = nn.Linear(self.input_dim, self.hidden_dim_1)
        self.rnn = nn.GRUCell(input_size=self.hidden_dim_1, hidden_size=self.hidden_dim_2)
        self.linear_out = nn.Linear(self.hidden_dim_2, self.output_dim)
        self.non_linear = nn.ReLU()

        self.init_func(self.linear_1.weight)
        self.init_func(self.rnn.weight_hh)
        self.init_func(self.rnn.weight_ih)
        self.init_func(self.linear_out.weight)

    def get_q_value(self, x, prev_hidden=None):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)

        x = self.non_linear(self.linear_1(x))
        hidden = self.rnn(x, prev_hidden)
        out = self.linear_out(hidden)
        return out, hidden.detach()
