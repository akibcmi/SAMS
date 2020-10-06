# -*- encoding=utf8 -*-

import os
import codecs


class EvalSaver(object):
    def openfile(self,epid,params,logger):
        return None

    def closefile(self,files:dict):
        if files is not None:
            for j in files:
                files[j].close()

    def writelog(self,message, logger):
        logger.info(message)



class SingleLabelEvalSaver(EvalSaver):
    def openfile(self,epochid,params,logger):
        savepath = params["savepath"]
        datafrom = params["datafrom"]
        curPath = os.getcwd()
        targetPath = curPath + os.path.sep + savepath
        if not os.path.exists(targetPath):
            os.makedirs(targetPath)

        fs = codecs.open(targetPath + os.path.sep + datafrom + "_" + str(epochid), "w", "utf8")
        self.writelog(targetPath + os.path.sep + datafrom + "_" + str(epochid),logger)
        return {"fs":fs}


class SegLayerEvalSaver(EvalSaver):
    def openfile(self,epochid,params,logger):
        savepath = params["savepath"]
        datafrom = params["datafrom"]
        curPath = os.getcwd()
        targetPath = curPath + os.path.sep + savepath
        if not os.path.exists(targetPath):
            os.makedirs(targetPath)

        fs = codecs.open(targetPath + os.path.sep + datafrom + "_" + str(epochid), "w", "utf8")
        self.writelog(targetPath + os.path.sep + datafrom + "_" + str(epochid), logger)
        fs2 = codecs.open(targetPath + os.path.sep + datafrom + "_" + str(epochid) + "_score6", "w", "utf8")
        self.writelog(targetPath + os.path.sep + datafrom + "_" + str(epochid)+ "_score6",logger)

        return {"fs":fs,"fs2":fs2}