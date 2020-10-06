# -*- encoding=utf8 -*-
import codecs

import torch
import torch.nn as nn
import numpy as np
import gensim
import codecs
import logging


class Vocabulary(object):

    def __init__(self,
                 pretrained_file = None,
                 dataset_file= None,
                 dims = 0,
                 train=True):

        self.pad="<PAD>"
        self.unk="<UNK>"
        self.bos="<S>"
        self.eos="</S>"

        self.wordset = set()

        self.pretrained_file = pretrained_file
        self.dataset_file = dataset_file
        self.id2char = dict()
        self.char2id = dict()
        self.word2vectors = dict()
        self.word2vectors_arr = None
        self.dims = dims

        self.train=train

        self.read_word2vec_file()

    @property
    def size(self):
        return self.id2char.size()


    def read_pretrain_file(self):
        word2vec_vector = dict()
        id2char = dict()
        char2id = dict()
        self.append_char("<PAD>", char2id, id2char, word2vec_vector, self.dims)
        self.append_char("<UNK>", char2id, id2char, word2vec_vector, self.dims)
        self.append_char("<S>", char2id, id2char, word2vec_vector, self.dims)
        self.append_char("</S>", char2id, id2char, word2vec_vector, self.dims)
        if self.pretrained_file == None or self.pretrained_file == "":
            return word2vec_vector, id2char, char2id, self.dims

        with codecs.open(self.pretrained_file , "r" , "utf8") as files:
            count_dims = files.readline()
            dims = int(count_dims.split()[1])
            count = 0
            for line in files.readlines():
                chars = line.strip("\n").strip().split()[0]
                line = line.strip("\n").strip().split()
                vectors = np.array(list(map(float , line[1:])))
                word2vec_vector[chars] = vectors
                ids = len(id2char)
                id2char[ids] = chars
                char2id[chars] = ids
                count += 1
                #lines.append(line)
            return word2vec_vector,id2char,char2id,dims

    def read_dataset_file(self):
        if self.train != True:
            return dict()
        if self.dataset_file ==None or self.dataset_file == "":
            return dict()
        else:
            char_dicts = dict()
            with codecs.open(self.dataset_file , "r" , "utf8") as files:
                for line in files.readlines():
                    line = line.strip("\n").strip()
                    lines = line.split()
                    for char in lines:
                        self.wordset.add(char)
                        char = list(char)
                        for chars in char:
                            char_dicts[chars] = chars
                    #for chars in line:
                        #char_dicts[chars] = chars

            return char_dicts

    def append_char(self,chars,char2id , id2char ,word2vec_vector,dims):
        if chars not in char2id:
            id2char[len(id2char)] = chars
            char2id[chars] = len(char2id)
        if chars not in word2vec_vector:
            word2vec_vector[chars] = np.random.rand(dims)

    def read_word2vec_file(self):
        if self.pretrained_file is None:
            print("No pretrained file")

        word2vec_vector,id2char,char2id,dims = self.read_pretrain_file()

        datasetdicts = self.read_dataset_file()

        for chars in datasetdicts:
            self.append_char(chars , char2id , id2char , word2vec_vector , dims)

        self.id2char = id2char
        self.char2id = char2id
        self.word2vectors = word2vec_vector

        vectors = list()
        for j in range(len(id2char)):
            vectors.append(word2vec_vector[id2char[j]])
        self.word2vectors_arr = np.array(vectors)


    def make_vocabs(self,lines , word2vec_vector,dims):
        self.id2char[0] = "<PAD>"
        self.id2char[1] = "<UNK>"
        self.id2char[2] = "<S>"
        self.id2char[3] = "</S>"

        self.char2id["<PAD>"] = 0
        self.char2id["<UNK>"] = 1
        self.char2id["<S>"] = 2
        self.char2id["</S>"] = 3

        for j in range(len(lines)):
            self.id2char[j+4] = lines[j]
            self.char2id[lines[j]] = j + 4

    def get_id(self,chars):
        if chars in self.char2id:
            return self.char2id[chars]
        else:
            return self.char2id["<UNK>"]
    def getchars(self,id):
        if id in self.id2char:
            return self.id2char[id]
        else:
            return "<UNK>"


    def source2batch(self,sentences):
        sentences = list(sentences.strip("\n").strip().split())
        target = []
        target.append(2)
        source = ""
        for words in sentences:
            words = words.strip("\t").strip().strip("\t")
            source = source + words
            if words == "":
                continue
            if len(words) == 1:
                target.append(2)
                continue
            arrs = [1 for j in range(len(words) - 1)]
            target = target + arrs
            target.append(2)
        sentences.append("</S>")
        sentences.insert(0,"<S>")
        return self.source2vector(source) , np.array(target,dtype=np.int64) , len(source)+2,len(target),source



    def source2vector(self,source):
        char_ids = [self.get_id(chars) for chars in list(source.strip("\n").strip())]
        char_ids.insert(0,2)
        char_ids.append(3)
        return np.array(char_ids,dtype=np.int64)

    def sentence2vector(self,sentences):
        char_ids = [self.get_id(chars) for chars in list(sentences.strip("\n").strip())]
        return np.array(char_ids , dtype=np.int64)

    def vector2sentence(self,vectors):
        chars_lists = [self.getchars(int(id)) for id in vectors]
        return " ".join(chars_lists)

class Dataset(object):
    def __init__(self , vocabs:Vocabulary , dataset_iters , batch_size = 32 , usebuffers = False):
        self.vocabs = vocabs
        self.dataset_iters = dataset_iters
        self.batch_size = batch_size
        self.useBuffers = usebuffers


    def __iter__(self):
        return self

    @property
    def word2vectors(self):
        return self.vocabs.word2vectors_arr

    def next(self):
        outs = []
        outs2 = []
        for ones in self.dataset_iters.next():

            yield ones["source"],ones["target"],ones["sentence"]

        raise StopIteration()

class BuffersDatasers(object):
    def __init__(self , data , batchSize = 1,tokens='sentence',cy=False,sort=False):
        self.data = data

        if sort:
            self.data.sort(key=lambda a : len("".join(a.strip("\n").strip().split())))

        self.endstarts = False
        self.batchSize = batchSize
        self.tokens=tokens
        self.cy=cy
        self.sort = sort
        self.begins = 0
        self.lengthsets = [len("".join(sens.strip("\n").strip().split())) for sens in data]
        self.lengthsusets = []
        self.lengthsusets.append(0)
        for j in range(1,len(self.lengthsets)):
            self.lengthsusets.append(self.lengthsusets[ j - 1] + self.lengthsets[j - 1])

    def restarts(self):
        self.endstarts = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.data is None or len(self.data) == 0:
            raise StopIteration()
        self.endstarts = False
        i = 0
        while i < len(self.data) and self.endstarts is False:
            les = i + self.batchSize
            if les > len(self.data):
                les = len(self.data)
            arrs = self.data[i:les]
            i = les
            yield arrs

        raise StopIteration()

    def bufferdata(self,starts,ends):
        if self.cy:
            if ends <= starts:
                return []
            else:
                startsto = starts % len(self.data)
                endsto = ends - starts + startsto
                gets = []
                leges = endsto - startsto
                if endsto > len(self.data):
                    gets = gets + self.data[startsto:]
                    cycou = (endsto - startsto) // len ( self.data )
                    for _ in range(cycou):
                        gets = gets + self.data
                    gets = gets + self.data[:endsto%len(self.data)]
                    return gets
                else:
                    return self.data[startsto:endsto]
        else:
            if starts >= len(self.data) or ends <= starts:
                return []
            else:
                return self.data[starts:len(self.data) if ends > len(self.data) else ends]
    def tokendata(self,starts):

        gesles=0
        j=starts
        ges = []
        while  j < len(self.data) and ( gesles + self.lengthsets[j] <= self.batchSize or  len(ges) == 0):
            gesles = gesles + self.lengthsets[j]
            ges.append(self.data[j])
            j = 1 + j
            if self.cy and j >= len(self.data):
                j = 0
        
        return ges ,j

    def next(self):
        if self.data is None or len(self.data) == 0:
            raise StopIteration()
        self.endstarts = False

        if self.tokens == "sentence":
            i = 0
            while (i < len(self.data) and self.endstarts is False) or self.cy :

                ges = self.bufferdata(i , i + self.batchSize)
                i = i + self.batchSize

                yield ges
        elif self.tokens=='token':
            i = 0
            arrs = []
            arrsl = 0
            while i < len(self.data) and self.endstarts is False:
                ges,i = self.tokendata(i)
                yield ges


        raise StopIteration()





class FilesReaderDataset(object):
    def __init__(self , filepaths , batch_size ,tokens='sentence',vocabs:Vocabulary = None, buffers=False ,buffersLarge=0 ,cy=False,sorts=True):

        self.files=filepaths
        self.batchSize = batch_size
        self.vocabs = vocabs
        self.endfiles = False
        self.buffers = buffers
        self.dataBuffers = None
        self.buffersLarges = buffersLarge
        self.tokens=tokens
        self.cy=cy
        self.sort=sorts
        self.bufferReaders = None
        self.datales = 0

        if self.buffers and self.buffersLarges != 0 and self.buffersLarges % self.batchSize != 0:
            raise Exception()
        self.preprocess()

    def preprocess(self):
        if self.buffers and self.cy:
            with codecs.open(self.files, "r", "utf8") as fs1:
                lines = fs1.readlines()
                self.dataBuffers = None
                self.datales = len(lines)
                self.bufferReaders = BuffersDatasers(lines , self.batchSize,self.tokens,self.cy,self.sort)
        else:
            if self.buffers and self.buffersLarges == 0:
                with codecs.open(self.files , "r" , "utf8") as fs1:
                    lines = fs1.readlines()
                    self.dataBuffers = []
                    self.datales = len(lines)
                    bufferDatas = BuffersDatasers(lines , self.batchSize,self.tokens,self.cy,self.sort)
                    buffers = []
                    for batchs in bufferDatas.next():
                        len(batchs)
                        batchs = self.preprocessBatch(batchs)
                        buffers.append(batchs)
                    self.dataBuffers.append(buffers)

    def __iter__(self):
        return self

    def restarts(self):
        self.endfiles = True

    def buildBuffers(self):
        if self.buffersLarges != 0:
            linecounts = 0
            with codecs.open(self.files , "r" , "utf8") as fs1:
                buffers = []
                batchs = []
                for line in fs1.readlines():
                    linecounts = linecounts + 1
                    batchs.append(line)
                    if len(batchs) == self.batchSize:
                        buffers.append(self.preprocessBatch(batchs))
                        batchs = []
                    if linecounts == self.buffersLarges:
                        linecounts = 0
                        yield buffers
                        buffers = []
                    if self.endfiles:
                        raise StopIteration()
                buffers.append(self.preprocessBatch(batchs))
                yield buffers
            raise StopIteration()
        else:
            for buffers in self.dataBuffers:
                if self.endfiles:
                    raise StopIteration()
                yield buffers
            raise StopIteration()

    def preprocessBatch(self , batchs):
        sourceBatch = []
        targetBatch = []
        sentences = []
        sourceMax = 0
        targetMax = 0
        for batch in batchs:
            source,target,sourceMaxlen,targetMaxlen,sentence = self.vocabs.source2batch(batch)
            sourceBatch.append(source)
            sentences.append(sentence)
            if sourceMaxlen > sourceMax:
                sourceMax = sourceMaxlen
            targetBatch.append(target)
            if targetMaxlen>targetMax:
                targetMax = targetMaxlen
        source = np.zeros((len(sourceBatch) , sourceMax),dtype=np.int64)
        target = np.zeros((len(targetBatch) , targetMax),dtype=np.float32)
        for idx , [sourcej,targetj] in enumerate(zip(sourceBatch,targetBatch)):
            source[idx , :sourcej.size] = sourcej
            target[idx , :targetj.size] = targetj

        batchs = {"source":source,"target":target,"targetlen":targetMax,"sourcelen":sourceMax,"sentence":sentences}

        return batchs


    def restarts(self):
        self.endfiles = True

    def next(self):
        for batchs in self.nextBatchs():
            yield batchs
            if self.endfiles:
                self.endfiles=False
                raise StopIteration()
        raise StopIteration()
    @property
    def datalengs(self):
        return self.datales

    def nextBatchs(self):

        if self.buffers:
            if self.cy:
                for batchs in self.bufferReaders.next():
                    yield self.preprocessBatch(batchs)
                    if self.endfiles:
                        raise StopIteration()
            else:
                for buffers in self.buildBuffers():
                    for batchs in buffers:
                        yield batchs
                        if self.endfiles:
                            raise StopIteration( )
                    if self.endfiles:
                        raise StopIteration()
            raise StopIteration()
        else:
            with codecs.open(self.files , "r","utf8") as fs1:
                batchs=[]
                for line in fs1.readlines():
                    batchs.append(line)
                    if len(batchs) == self.batchSize:
                        batchs = self.preprocessBatch(batchs)
                        yield batchs
                        if self.endfiles:
                            raise StopIteration()
                        batchs = []
                raise StopIteration( )













