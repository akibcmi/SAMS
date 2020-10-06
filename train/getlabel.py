#-*- encoding=utf8 -*-

import numpy as np
import codecs
import torch
import torch.nn as nn
from .prf1 import CalculatePRF
from data import Vocabulary

class SeqLabelResult(object):
    def decode(self, input, sen, target, files:dict, cous:dict,prf:CalculatePRF,vocabs:Vocabulary):
        pass


class CNSegSeqLabelResult(SeqLabelResult):
    def decode(self, input, sens, target, files:dict, cous:dict,prf:CalculatePRF,vocabs:Vocabulary):
        files = files["fs"]
        outs = torch.argmax(input, 2)
        outs = np.array(outs.data.cpu())[:, 1:]

        cous,sens = prf.calculate_prf(outs,target,sens,cous,vocabs)
        for s in sens:
            self.write_text(s,files)
            self.write_text("\n",files)
        return cous

    def write_text(self, text, files):
        files.write(text)
        return files

class CNSeglayerLabelResult(CNSegSeqLabelResult):
    def decode(self, input, sens, target, files:dict, cous:dict,prf:CalculatePRF,vocabs:Vocabulary):
        fs = files["fs"]
        fs2 = files["fs2"]
        out,out2 = input

        cous = super().decode(out,sens,target, {"fs":fs},cous,prf,vocabs)
        if out2 is not None and fs2 is not None:
            super().decode(out2,sens, target,{"fs":fs2},{
            "countright":0,
            "countpred":0,
            "countgold":0,
            "ovright":0,
            "ovgold":0,
            "inright":0,
            "ingold":0
        },prf,vocabs)
        return cous


