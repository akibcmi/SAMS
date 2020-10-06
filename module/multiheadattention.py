# -*- encoding=utf8 -*-

import torch
import torch.nn as nn
import numpy as np
import math
from scipy.stats import norm



class MultiHeadAttention(nn.Module):

    def __init__(self , dims , heads , dropout,window=5,type="center"):
        super(MultiHeadAttention , self).__init__()
        self.dims = dims
        self.dims_heads = dims // heads
        assert dims % heads == 0
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.linearkey = nn.Linear(dims , dims)
        self.linearvalue = nn.Linear(dims , dims)
        self.linearquery = nn.Linear(dims , dims)
        self.softmax = nn.Softmax(dim=-1)
        self.final_linear = nn.Linear(dims , dims)
        self.window=0
        self.types=type


        def normal_distribution(x,means,squares):
            return norm(means,squares).cdf(x) *2
        if self.types == "forward" or self.types == "backward":
            squares = 4
        else:
            squares = 4
        maxlens =1500
        self.windowmax = torch.zeros(maxlens,maxlens)
        if self.window > 0 :
            windowmasks = torch.zeros(maxlens, maxlens)
            for j in range(self.window):
                len = maxlens - j
                masknorm = normal_distribution(-j, 0, squares)
                ones = torch.ones(len) * masknorm
                ones = torch.diag(ones, j)
                windowmasks = windowmasks + ones
            self.windowmax = windowmasks
            self.windowmax = self.windowmax + self.windowmax.transpose(0, 1) - torch.diag(torch.ones(maxlens))
        else:
            windowmasks = torch.zeros(maxlens, maxlens)
            for j in range(maxlens):
                len = maxlens - j
                masknorm = normal_distribution(-j, 0, squares)
                ones = torch.ones(len) * masknorm
                ones = torch.diag(ones, j)
                windowmasks = windowmasks + ones
            self.windowmax = windowmasks
            self.windowmax = self.windowmax + self.windowmax.transpose(0, 1) - torch.diag(torch.ones(maxlens))
        window = torch.ones(maxlens , maxlens , dtype=torch.uint8)
        if self.window > 0:
            window_masks = torch.tril(window,diagonal=self.window)
            window_masks = window_masks - torch.tril(window,diagonal=-self.window)
            if self.types == "forward":
                window_masks = torch.tril(window_masks)
            elif self.types == "backward":
                windows_forward = torch.tril(window_masks,diagonal=-1)
                window_masks = window_masks - windows_forward
            window_masks = window - window_masks
        else:
            window_masks = torch.tril(window, diagonal=maxlens)
            window_masks = window_masks - torch.tril(window, diagonal=-maxlens)
            if self.types == "forward":
                window_masks = torch.tril(window_masks)
            elif self.types == "backward":
                windows_forward = torch.tril(window_masks, diagonal=-1)
                window_masks = window_masks - windows_forward
            window_masks = window - window_masks
        self.window_masks = window_masks

        if torch.cuda.is_available():
            self.windowmax=self.windowmax.cuda()
            self.window_masks = self.window_masks.cuda()


    def forward(self, key,value,query,mask = None):
        batchSize = key.size(0)
        key_len = key.size(1)

        key = self.linearkey(key)
        value = self.linearvalue(value)
        query = self.linearquery(query)



        windowsmasknorm = self.windowmax[:key_len,:key_len]

        window_masks = self.window_masks[:key_len,: key_len]


 

        

        if torch.cuda.is_available():
            window_masks = window_masks.cuda()
            windowsmasknorm = windowsmasknorm.cuda()
        windowsmasknorm = windowsmasknorm.masked_fill( window_masks, 1)
        window_masks = window_masks.unsqueeze(0).unsqueeze(0)
        def shapes(x):
            return x.view(batchSize , -1 , self.heads , self.dims_heads).transpose(1,2)

        def reshapes(x):
            return x.transpose(1,2).contiguous().view(batchSize , -1 , self.dims)



        key = shapes(key)
        value = shapes(value)
        query = shapes(query)

        query = query / math.sqrt(self.dims_heads)
        scores = torch.matmul(query , key.transpose(2,3))

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask,-1e18)
            scores = scores.masked_fill(window_masks,-1e18)
            windowsmasknorm = windowsmasknorm.unsqueeze(0).unsqueeze(0)
            scores = scores * windowsmasknorm


        attn = self.softmax(scores)
        if mask is not None:
            attn = attn.masked_fill(mask,0)
        drop_atten = self.dropout(attn)
        context = reshapes(torch.matmul(drop_atten , value))

        output = self.final_linear(context)

        return output






