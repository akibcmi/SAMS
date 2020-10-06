# -*- encoding=utf8 -*-

import torch
import torch.nn as nn



class CalculateLoss(object):
    def calculateloss(self, out, target, criterion):
        return None


class SingleLabelLoss(CalculateLoss):
    def calculateloss(self, outs, target, criterion):
        size = list(target.shape)
        loss = criterion(outs.view(size[0] * size[1], -1), target.view(-1))
        return loss


class SegLayerLoss(CalculateLoss):
    def calculateloss(self, outs, target, criterion):
        outs1,outs2 = outs
        size = list(target.shape)
        loss = criterion(outs1.view(size[0] * size[1], -1), target.view(-1))
        if outs2 is not None:
            loss2 = criterion(outs2.view(size[0] * size[1], -1), target.view(-1))
            loss = loss + loss2
        return loss
