# -*- encoding = utf8 -*-
from data import Vocabulary,Dataset,FilesReaderDataset
import logging
import os
import time
from torch.autograd import Variable
from .optimizers import Optimizer

import torch
from torch import nn as nn
import numpy as np
import torch.functional as Functional
import codecs
import subprocess
from .evalsaver import EvalSaver
from .callos import CalculateLoss
from .prf1 import CalculatePRF
from .getlabel import SeqLabelResult



class Trainer(object):

    def __init__(self,
                 trainData:FilesReaderDataset,
                 validData:FilesReaderDataset,
                 vocabs:Vocabulary,
                 optimizer:Optimizer,
                 criterion,
                 saves,
                 epoch,
                 batch,
                 model,
                 datafrom,
                 shows=100,
                 logger=None,
                 gpu = True,
                 modelfiles = None,
                 steps=0,
                 epochsteps=0,
                 savepath=None,
                 cy=False,
                 validsteps=0,
                 trainsteps=0,
                 savesteps =1,
                 accumulate =1,
                 name=None,
                 dataset=None,
                 getlabel:SeqLabelResult=None,
                 calloss:CalculateLoss=None,
                 evalsaver:EvalSaver=None
                 ):
        self.trainData = trainData
        self.validData = validData
        self.vocabs = vocabs
        self.saves = saves
        self.epoch = epoch
        self.batchSize = batch
        self.model = model
        self.gpu=gpu
        self.datafrom = datafrom
        self.modelfiles = modelfiles
        self.optimizer=optimizer
        self.criterion=criterion
        self.shows = shows
        self.logs = logger
        self.steps = steps
        self.epochssteps=epochsteps
        self.savepath=savepath
        self.cy=cy
        self.trainsteps = trainsteps
        self.validsteps=validsteps
        self.savesteps = savesteps
        self.accumulate = accumulate
        self.accumulatesteps = 0
        self.dataset = dataset
        self.name = name
        self.maxjvalues = 0.0
        self.getlabel = getlabel
        self.calloss = calloss
        if calloss is None:
            raise Exception("No loss")
        self.evalsaver = evalsaver

        if self.cy:
            assert self.trainsteps > 0
            assert self.validsteps > 0
            assert self.savesteps > 0


        self.trainBatchs = Dataset(self.vocabs , trainData , batch)
        self.validBatchs = Dataset(self.vocabs , validData , batch)

        self.prf1 = CalculatePRF()
    def write_logger(self,message):
        self.logs.info(message)

    def train(self):
        self.write_logger("Training begin")
        self.optimizer.zero_grad()
        if self.cy:
            self.write_logger("Training with cy, so the trains steps open")
            self.write_logger("Trainsteps %d" %(self.trainsteps))
            self.write_logger("Validsteps %d" %(self.validsteps))
            assert self.trainsteps > 0 and self.validsteps > 0
            datales = self.trainData.datalengs
            allsles=0
            loss_steps=[]
            starts=time.time()
            epoch=0
            self.write_logger("Epoch %d" % (epoch))
            self.write_logger("Train epoch %d" % (epoch))
            self.write_logger("Train step %d" % (self.steps))
            for steps,batchs in zip(range(self.steps , self.trainsteps),self.trainBatchs.next()):
                singleles,loss_steps,starts = self.train_steps(steps,batchs,loss_steps,epoch,starts)
                allsles = allsles + singleles
                if ( steps - self.steps ) % self.validsteps == 0 and steps != self.steps:
                    self.write_logger("Valid epoch %d" % (epoch))
                    self.write_logger("Valid step %d" % (steps))
                    cous = self.valid_steps(steps)
                if (steps - self.steps) % self.savesteps == 0 and steps != self.steps:
                    self.saves.saves(steps,epoch)
                if allsles > datales:
                    allsles - datales

                    if ( steps - self.steps ) % self.validsteps != 0:
                        self.write_logger("Valid epoch %d" % (epoch))
                        cous = self.valid_once(epoch)
                    if (steps - self.steps) % self.savesteps != 0 :
                        self.saves.saves(steps, epoch)
                    epoch = epoch + 1
                    self.write_logger("Epoch %d" % (epoch))
                    self.write_logger("Train epoch %d" % (epoch))
                    self.write_logger("Train step %d" % (steps))

        else:
            for uid in range(self.epochssteps,self.epoch):
                self.write_logger("Epoch %d" % (uid))
                self.write_logger("Train epoch %d" % (uid))
                self.train_once(uid)
                self.write_logger("Valid epoch %d" % (uid))
                cous = self.valid_once(uid)
                jvalues = cous['f1']
                if jvalues < 0:
                    self.savecheckpoints(uid)
                elif jvalues > self.maxjvalues:
                    self.maxjvalues = jvalues
                    self.savecheckpoints(uid,"best")
                    self.write_logger("Epoch "+str(uid)+" best " + str(jvalues))
                else:
                    self.write_logger("Epoch "+str(uid)+" "+ str(jvalues))
                self.write_logger("Epoch %d finished" % (uid))

    def train_steps(self,steps,train_batchs,loss_steps,epochs,starts):


        target = train_batchs[1]
        les = len(target)
        source = train_batchs[0]
        source = Variable(torch.Tensor(source).long()).contiguous()
        target = Variable(torch.Tensor(target).long()).contiguous()

        if self.gpu:
            source = source.cuda()
            target = target.cuda()


        self.model.train()
        self.criterion.train()


        outs = self.model(source)
        loss = self.calloss.calculateloss(outs,target, self.criterion)
        loss.backward()
        if self.accumulatesteps % self.accumulate == 0 and self.accumulatesteps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.accumulatesteps = 1 + self.accumulatesteps
        usetimes = time.time() - starts
        loss_oce = loss.data.item()
        loss_steps.append(np.average(np.array(loss_oce)))
        if steps % self.shows == 0 and steps is not 0:
            cost = np.average(np.array(loss_steps))
            self.write_logger("Epoch {0} Batch {1} Cost {2} UseTime {3} LR {4} Steps {5}".format(epochs, steps, cost, usetimes,
                                                                                       self.optimizer.learning_rate,self.optimizer._step))
            starts = time.time()
            loss_epoch = []
        del loss, source, target, outs, loss_oce
        torch.cuda.empty_cache()
        return les,loss_steps,starts

    def valid_steps(self,steps):
        if self.evalsaver is None:
            self.write_logger("No Eval")
            return

        fs = self.evalsaver.openfile(steps, {"datafrom": self.datafrom, "savepath": self.savepath}, self.logs)
        cous = self.valid(fs)
        if fs is not None:
            self.evalsaver.closefile(fs)
        self.write_logger("Valid finished")
        return cous

    def train_once(self,epochid):
        loss_epoch=[]
        showsid = 0
        #self.optimizer.zero_grad()
        self.model.train()
        self.criterion.train()
        starts = time.time()
        for uid,train_batchs in enumerate(self.trainBatchs.next()):
            try:
                target = train_batchs[1]
                source = train_batchs[0]
                source = Variable(torch.Tensor(source).long()).contiguous()
                target = Variable(torch.Tensor(target).long()).contiguous()
            
                if self.gpu:
                    source = source.cuda()
                    target=target.cuda()

                outs = self.model(source)

                loss = self.calloss.calculateloss(outs,target,self.criterion)
                loss.backward()

                if self.accumulatesteps % self.accumulate == 0 and self.accumulatesteps != 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.accumulatesteps = 1 + self.accumulatesteps

                usetimes = time.time()-starts

                loss_oce = loss.data.item()
                loss_epoch.append(np.average(np.array(loss_oce)))

                if uid % self.shows == 0 and uid is not 0:
                    cost = np.average(np.array(loss_epoch))
                    self.write_logger("Epoch {0} Batch {1} Cost {2} UseTime {3} LR {4} Step {5}".format(epochid,uid,cost,usetimes,self.optimizer.learning_rate,self.optimizer._step))
                    starts = time.time()
                    loss_epoch = []

                del loss, source,target,outs,loss_oce
                torch.cuda.empty_cache()
            except Exception as e:
                print(e)
            
        self.write_logger("Epoch {0} Cost {1} UseTime{2} LR {3} Step {4}".format(epochid,np.average(np.array(loss_epoch)),time.time()-starts,self.optimizer.learning_rate,self.optimizer._step))
        self.write_logger("Save model")
        self.saves.saves(0,epochid,"last")

    def valid_once(self,epochid):
        if self.evalsaver is None:
            self.write_logger("No Eval")
            return

        fs = self.evalsaver.openfile(epochid,{"datafrom":self.datafrom,"savepath":self.savepath},self.logs)
        cous = self.valid(fs)
        if fs is not None:
            self.evalsaver.closefile(fs)
        self.write_logger("Valid finished")
        return cous

    def valid(self,fs):
        self.model.eval()
        with torch.no_grad():
            cous = self.prf1.initprf()
            for uid, train_batchs in enumerate(self.validBatchs.next()):
                starts = time.time()
                source = train_batchs[0]
                sens = train_batchs[2]
                source = Variable(torch.Tensor(source).long()).contiguous()
                if self.gpu:
                    source = source.cuda()

                outs = self.model(source)
                if self.getlabel is not None:
                    cous = self.getlabel.decode(outs,sens,train_batchs[1],fs,cous,self.prf1,self.vocabs)
            cous = self.prf1.get_score(cous)
            self.write_logger("P: {0}, R: {1}, F1: {2}".format(cous['p'],cous['r'],cous['f1']))
            self.write_logger(cous)
            return cous

    def savecheckpoints(self,uid,name=None):
        self.model.train()
        self.criterion.train()
        self.saves.saves(0, uid,name)









