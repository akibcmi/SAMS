# -*- encoding=utf-8 -*-

import argparse
import time
import numpy as np

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
from module import EmbeddingPE
import math
import codecs
from train import SegLayerEvalSaver
from train import SegLayerLoss
from train import EvalSaver
from train import CNSeglayerLabelResult


def create_parse_valid():
    parser = argparse.ArgumentParser()
    # trainfiles
    parser.add_argument("--logfiles", type=str, default=""
                        , help="path to log files")
    parser.add_argument("--name", type=str, required=True
                        , help="name of dataset")
    parser.add_argument("--evalfile", type=str, required=True)
    parser.add_argument("--savefiles", type=str, required=True)
    
    parser.add_argument("--showsteps", type=int, default=100)
    parser.add_argument("--gpu", action="store_true")

    parser.add_argument("--loss", default="crossentropyloss", choices=['crossentropyloss', 'nllloss'])
    parser.add_argument("--valid_token", default='sentence', choices=['token', "sentence"])
    parser.add_argument("--valid_batch_size", default=32, type=int)
    parser.add_argument("--use_buffers",action="store_true")
    parser.add_argument("--buffer_size",type=int,default=0)

    parser.add_argument("--model",type=str,required=True)
    return parser.parse_args()



def create_vocabs(opts,checkpointer):
    vocabs = None
    vocabs = checkpointer["vocabs"]
    return vocabs


def create_embeddings(opts,train_ops,vocabs,checkpointer):
    pretrain_embeddings = Variable(torch.Tensor(vocabs.word2vectors_arr))
    embeddings = nn.Embedding.from_pretrained(pretrain_embeddings)
    embeddings = EmbeddingPE(embeddings,train_ops.dropout,train_ops.dim,train_ops.position_encoding)

    embeddings.load_state_dict(checkpointer["embeddings"])

    if opts.gpu and torch.cuda.is_available():
        embeddings = embeddings.cuda()

    return embeddings


def create_valid_dataset(opts , vocabs,logger):
    dataset = FilesReaderDataset(opts.evalfile,opts.valid_batch_size,
                                 opts.valid_token,vocabs,opts.use_buffers,
                                 opts.buffer_size,False,False)
    logger.info("Dataset from %s, batch_size: %d" % (opts.evalfile, opts.valid_batch_size))
    return dataset

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

def create_checkpointer(opts , logger):
    if opts.model:
        logger.info("Loading checkpoint from %s" % opts.model)
        checkpointer = torch.load(opts.model)
        train_ops = checkpointer["opts"]
        return checkpointer,train_ops
    else:
        return None

def create_models(opts,train_opts,embeddings,checkpointer,logger):
    model = BiaffineSegmentationModel(train_opts.dim,train_opts.layer,train_opts.head,train_opts.ff,train_opts.dropout,embeddings,train_opts.window,train_opts.norm_after,train_opts.seglayers)

    model.load_state_dict(checkpointer["model"],strict=False)
    if opts.gpu and torch.cuda.is_available():
        model = model.cuda()
    return model

def create_getlabel(opts):
    return CNSeglayerLabelResult()

def create_calloss(opts):
    return SegLayerLoss()

def create_eval(opts):
    return SegLayerEvalSaver()

def valid_single():
    opts = create_parse_valid()

    #createlogs
    logger = create_logs(opts)

    logger.info("Start creating.")

    for k in opts.__dict__:
        logger.info(k + ":" + str(opts.__dict__[k]))


    #create checkpointer
    logger.info("Checkpoint.")
    checkpoint,train_ops = create_checkpointer(opts , logger)

    #create vocabs
    logger.info("Vocabs")
    vocabs = create_vocabs(opts,checkpoint)

    #create embeddings
    logger.info("Embeddings")
    if train_ops.position_encoding:
        logger.info("Use Position Encoding")
    embeddings = create_embeddings(opts,train_ops,vocabs,checkpoint)

    #create models
    logger.info("Create Models")
    model = create_models(opts,train_ops,embeddings,checkpoint,logger)


    logger.info("Valid dataset")
    validdataset = create_valid_dataset(opts , vocabs,logger)

    logger.info("Getlabel")
    getlabel = create_getlabel(opts)

    logger.info("Calclualte Loss")
    calloss = create_calloss(opts)

    logger.info("Eval Files")
    evalsaver = create_eval(opts)

    sofs = nn.Softmax(-1)
    try:
        logger.info("Valid start.")
        logger.info("Model from " + opts.model)
        validBatchs = Dataset(vocabs, validdataset, opts.valid_batch_size)
        model.eval()
        with torch.no_grad():
            score2 = ""
            sens2 = ""
            startvalid = time.time()
            starts = time.time()
            for uid, train_batchs in enumerate(validBatchs.next()):
                starts = time.time()
                source = train_batchs[0]
                sens = train_batchs[2]
                source = Variable(torch.Tensor(source).long()).contiguous()
                if opts.gpu:
                    source = source.cuda()

                outs,out2 = model(source)
                outscore = outs[:,:,:]
                outscore = sofs(outscore)
                outscore = np.array(outscore.data.cpu())[:,:,:]
                outs = torch.argmax(outs, 2)
                outs = np.array(outs.data.cpu())[:, 1:]
                
                usetimes = time.time() - starts
                for j in range(outs.shape[0]):
                    sentence = sens[j]
                    out = outs[j]
                    scorej = outscore[j]
                    sen = ""
                    scores = ""
                    for idx, char in enumerate(sentence):
                        sen += char
                        scores = scores + str(scorej[idx,2])
                        scores = scores + " "
                        if idx < len(out) and idx != len(sentence) - 1 and out[idx] == 2:
                            sen += " "
                    sens2 = sens2 + sen
                    sens2 = sens2 + "\n"
                    score2 = score2 + scores
                    score2 = score2 + "\n"
                if uid % opts.showsteps == 0 and uid is not 0:
                    logger.info("Steps {0} Cost {1}".format(str(uid),str(time.time() - starts)))
                    starts = time.time()
            logger.info("Valid finish")
            logger.info("Valid cost " + str(time.time() - startvalid))
            logger.info("Valid saves " + opts.savefiles)
            with codecs.open(opts.savefiles , "w" , "utf8") as fs:
                fs.write(sens2)
    except Exception as e:
        ms = traceback.format_exc()
        logger.info(ms)


if __name__=="__main__":
    valid_single()
