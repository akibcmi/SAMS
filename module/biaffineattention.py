# -*- encoding=utf8 -*-
import torch
import torch.nn as nn

class BiaffineAttention(nn.Module):

    def __init__(self , dim , out_channels ,dropout, bias_heads = True , bias_dep=True):
        super(BiaffineAttention,self).__init__()

        self.linear1 = nn.Linear(dim , dim)
        self.linear2 = nn.Linear(dim , dim)
        self.bilinear = nn.Bilinear(dim , dim , out_channels)
        self.dropout = nn.Dropout(dropout)
        self.dropout_cat = nn.Dropout(dropout)
        self.linear = nn.Linear(dim * 2 , out_channels)

    def forward(self, h , d):
        h = self.dropout(self.linear1(h))  
        d = self.dropout(self.linear2(d))

        s1 = self.bilinear(h,d)
        s2 = self.linear(torch.cat([h,d],2))

        return s1 + s2
