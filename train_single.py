# -*- encoding=utf-8 -*-

import argparse

import traceback
import torch
import torch.nn as nn
import logging
from torch.autograd import Variable
from model import BiaffineSegmentationModel

from data import Vocabulary,Dataset,FilesReaderDataset
from train import Trainer
from train import ModelSaver
from train import Optimizer
from train import create_parse
from train import SeqLabelResult
from train import CNSegSeqLabelResult
from train import CNSeglayerLabelResult
from train import CalculateLoss
from train import SingleLabelLoss
from train import SegLayerLoss
from train import EvalSaver
from train import SegLayerEvalSaver
from module import EmbeddingPE




def create_vocabs(opts,checkpointer):
    vocabs = None #Vocabulary(opts.pretrain_files , opts.trainfile,opts.dim,True)
    if checkpointer is not None:
        vocabs = checkpointer["vocabs"]
    else:
        vocabs = Vocabulary(opts.pretrain_files, opts.trainfile, opts.dim, True)
    return vocabs


def create_embeddings(opts,vocabs,checkpointer):
    #embeddings = nn.Embedding(vocabs.size,vocabs.dims)
    pretrain_embeddings = Variable(torch.Tensor(vocabs.word2vectors_arr))
    embeddings = nn.Embedding.from_pretrained(pretrain_embeddings)
    embeddings = EmbeddingPE(embeddings,opts.dropout,opts.dim,opts.position_encoding)
    if checkpointer is not None:
        embeddings.load_state_dict(checkpointer["embeddings"])

    if opts.gpu and torch.cuda.is_available():
        embeddings = embeddings.cuda()

    return embeddings

def create_train_dataset(opts , vocabs,logger):
    dataset = FilesReaderDataset(opts.trainfile,opts.batch_size,opts.token,
                                 vocabs,opts.use_buffers,
                                 opts.buffer_size,opts.cyc,opts.des)

    logger.info("Dataset from %s, batch_size: %d" % (opts.trainfile,opts.batch_size))
    return dataset

def create_valid_dataset(opts , vocabs,logger):
    dataset = FilesReaderDataset(opts.evalfile,opts.valid_batch_size,
                                 opts.valid_token,vocabs,opts.use_buffers,
                                 opts.buffer_size,False,False)
    logger.info("Dataset from %s, batch_size: %d" % (opts.evalfile, opts.valid_batch_size))
    return dataset



def create_optim(opts,checkpointer,model):
    optim = None
    if checkpointer is not None:
        optim= checkpointer["optim"]
    else:
        optim= Optimizer(opts.optim,
                          opts.learning_rate,
                          0,
                          lr_decay=0.5,
                          beta1=opts.adam_beta1,
                          beta2=opts.adam_beta2,
                          decay_method=opts.decay_method,
                          start_decay_steps=opts.start_decay_steps,
                          decay_steps=opts.decay_steps,
                          warmup_steps=opts.warmup_steps,
                          model_size=opts.dim,
                         warmup_start_lr=opts.warmup_start_lr,
                         optims=opts.optims
                         )
    if opts.reloadlrs:
        opts.reloadlrs = False
        optim.reloadlrs(opts.learning_rate)
    #optim.method="adamax"
    optim.set_parameters(model)
    print(optim.optimizer)
    return optim


def create_saver(opts,checkpointer,model,optim,vocabs,embeddings,logger):
    saver = ModelSaver(opts.savefiles ,model,opts,optim,opts.savesteps,vocabs,embeddings,logger)
    return saver

import os
import logging
import sys

def init_logger(log_file = None ):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='a+')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger




class Logger(object):
    def __init__(self,logger):
        self.logger = logger

    def write(self,message):
        self.logger.info(message)

    def info(self,message):
        self.write(message)

    def flush(self):
        pass


def create_logs(opts):

    logger = Logger(init_logger(opts.logfiles))
    sys.stdout = logger
    return logger

def create_criterion(opts):
    if opts.loss == "crossentropyloss":
        return nn.CrossEntropyLoss(ignore_index=0)
    elif opts.loss == "nllloss":
        return nn.NLLLoss(ignore_index=0,reduction="sum")

def create_getlabel(opts):
    return CNSeglayerLabelResult()

def create_calloss(opts):
    return SegLayerLoss()

def create_eval(opts):
    return SegLayerEvalSaver()

def create_trainer(opts,checkpointer,traindataset,validdataset,vocab,model,saver,optim,logger,getlabel,calloss,evalsaver):
    criterion = create_criterion(opts)
    if opts.gpu:
        criterion = criterion.cuda()
    steps = 0
    if checkpointer is None:
        steps = 0
        epoch = 0
    else:
        steps = checkpointer["steps"] + 1
        epoch = checkpointer["epoch"]  +1
    trainer = Trainer(traindataset,
                      validdataset,
                      vocab,
                      optim,
                      criterion ,
                      saver,
                      opts.epoch,
                      opts.batch_size,
                      model,
                      opts.name,
                      opts.showsteps,
                      logger,
                      opts.gpu,
                      opts.train_from,
                      steps,
                      epoch,
                      opts.savevalid,
                      opts.cyc,
                      opts.valid_steps,
                      opts.train_steps,
                      opts.savesteps,
                      opts.accumulate,
                      opts.name,
                      opts.dataset,
                      getlabel,
                      calloss,
                      evalsaver)
    return trainer

def create_checkpointer(opts , logger):
    if opts.train_from:
        logger.info("Loading checkpoint from %s" % opts.train_from)
        checkpointer = torch.load(opts.train_from)
        epoch =  opts.epoch
        reloadlrs = opts.reloadlrs
        l = opts.learning_rate
        multi = opts.multigpu
        accumulate = opts.accumulate
        batch_size = opts.batch_size
        opts.__dict__.update(checkpointer["opts"].__dict__)
        opts.epoch = epoch
        opts.batch_size = batch_size
        opts.accumulate = accumulate
        opts.multigpu=multi
        if reloadlrs:
            opts.learning_rate = l
            opts.reloadlrs = True
        else:
            opts.reloadlrs = False
        return checkpointer
    else:
        return None

def create_models(opts,embeddings,checkpointer,logger):
    model = BiaffineSegmentationModel(opts.dim,
                                      opts.layer,
                                      opts.head,
                                      opts.ff,
                                      opts.dropout,
                                      embeddings,
                                      opts.window,
                                      opts.norm_after,
                                      opts.seglayers,
                                      opts.segwords,
                                      opts.middecode,
                                      opts.gate)

    if checkpointer is not None:
        model.load_state_dict(checkpointer["model"],strict=False)
    if opts.gpu and torch.cuda.is_available():
        model = model.cuda()

    if opts.multigpu:
        model=nn.DataParallel(model)
    logger.info(model)
    paracous = sum(x.numel() for x in model.parameters())
    logger.info("number of parameters is :" + str(paracous) )
    return model

def train_single():
    opts = create_parse()

    #createlogs
    logger = create_logs(opts)

    logger.info("Start creating.")

    for k in opts.__dict__:
        logger.info(k + ":" + str(opts.__dict__[k]))


    #create checkpointer
    logger.info("Checkpoint.")
    checkpoint = create_checkpointer(opts , logger)

    #create vocabs
    logger.info("Vocabs")
    vocabs = create_vocabs(opts,checkpoint)

    #create embeddings
    logger.info("Embeddings")
    if opts.position_encoding:
        logger.info("Use Position Encoding")
    embeddings = create_embeddings(opts,vocabs,checkpoint)

    #create models
    logger.info("Create Models")
    model = create_models(opts,embeddings,checkpoint,logger)

    #create optim
    logger.info("Optim")
    optims = create_optim(opts,checkpoint,model)

    #savers
    logger.info("Saver")
    savers = create_saver(opts,checkpoint,model,optims,vocabs,embeddings,logger)

    #traindataset
    logger.info("Train dataset")
    traindataset = create_train_dataset(opts , vocabs,logger)

    logger.info("Valid dataset")
    validdataset = create_valid_dataset(opts , vocabs,logger)

    logger.info("Getlabel")
    getlabel = create_getlabel(opts)

    logger.info("Calclualte Loss")
    calloss = create_calloss(opts)

    logger.info("Eval Files")
    evalsaver = create_eval(opts)

    #create trainer
    logger.info("Trainer")
    trainer = create_trainer(opts,checkpoint,traindataset,validdataset,vocabs,model,savers,optims,logger,getlabel,calloss,evalsaver)
    try:
        trainer.train()
    except Exception as e:
        ms = traceback.format_exc()
        logger.info(ms)


if __name__=="__main__":
    train_single()
