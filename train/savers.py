# -*- encoding=utf-8 -*-

import os
import torch
import torch.nn as nn


from .optimizers import Optimizer

class ModelSaver(object):

    def __init__(self,basepath , model:nn.Module, opts , optim:Optimizer , savesteps,vocab,embeddings:nn.Embedding,logger):
        self.basepath = basepath
        self.model  = model
        self.opts = opts
        self.optim=optim
        self.vocab = vocab
        self.savesteps = savesteps
        self.embeddings = embeddings
        self.logger = logger


    def saves(self,steps,epoch,name=None):
        if isinstance(self.model,nn.DataParallel):
            model_state=self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        checkpointer = {
            'model':model_state,
            'vocabs':self.vocab,
            'optim':self.optim,
            'steps':steps,
            'epoch':epoch,
            'embeddings':self.embeddings.state_dict(),
            'opts':self.opts
        }
        if name is None:
            checkpointerpath = "%s_epoch_%d_step_%d.pt" % (self.basepath,epoch,steps)
        else:
            checkpointerpath = "%s_epoch_%s_step_%d.pt" % (self.basepath,name,steps)
        self.logger.info("Save checkpointer %s" % (checkpointerpath))
        torch.save(checkpointer,checkpointerpath)
        self.logger.info("Finish")
        return checkpointer,checkpointerpath

