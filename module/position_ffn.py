# -*- encoding=utf8 -*-

import  torch
import torch.nn as nn
import math


class PositionwiseFeedForward(nn.Module):

    def __init__(self, dims , d_ff , dropout ,norm_after = False):
        super(PositionwiseFeedForward , self).__init__()
        self.w_1 = nn.Linear(dims , d_ff)
        self.w_2 = nn.Linear( d_ff , dims)
        self.layer_norm = nn.LayerNorm(dims , eps = 1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_after=norm_after

    def forward(self,x):
        if self.norm_after:
            inter = self.dropout_1(self.relu(self.w_1(x)))
            output = self.dropout_2(self.w_2(inter)) + x
            return self.layer_norm(output)
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x
