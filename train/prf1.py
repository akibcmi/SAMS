#-*-encoding=utf8-*-

from data import Vocabulary


class CalculatePRF(object):
    def write_logger(message,log):
        if log is not None:
            log.info(message)
    def initprf(self):
        return {
            "countright":0,
            "countpred":0,
            "countgold":0,
            "ovright":0,
            "ovgold":0,
            "inright":0,
            "ingold":0
        }

    def calculate_prf(self,preds,targets,sentence, cous : dict,
                      vocabs:Vocabulary=None,logger=None):
        senslen = len(preds)
        assert len(preds) == len(targets)

        countgold = cous["countgold"]
        countright = cous["countright"]
        countpred = cous["countpred"]

        inright = cous["inright"]
        ingold = cous["ingold"]

        ovright = cous["ovright"]
        ovgo:ld = cous["ovgold"]

        sens = []

        for j in range(senslen):
            pred = preds[j]
            target = targets[j]
            ses = sentence[j]

            inword = True
            predword = ""
            goldword = ""
            seso = ""
            for l in range(len(ses)):

                singlep = pred[l]
                singleg = target[l+1]
                predword = predword + ses[l]
                goldword = goldword + ses[l]

                if l == len(ses)-1:
                    singlep = 2
                    singleg = 2

                if singlep == 2:
                    countpred = countpred + 1
                    seso = seso + " " + predword
                    predword = ""
                if singleg == 2:
                    countgold = countgold + 1
                    if goldword in vocabs.wordset:
                        ingold =  ingold + 1
                    else:
                        ovgold = ovgold + 1
                    ovword = goldword
                    goldword = ""

                if singlep == singleg:
                    if singlep == 2:
                        if inword:
                            countright = countright + 1
                            if ovword in vocabs.wordset:
                                inright = inright + 1
                            else:
                                ovright = 1 + ovright
                        inword = True
                    else:
                        continue
                if singlep != singleg:
                    inword = False

            seso = seso + " " + predword
            seso = seso.strip()
            sens.append(seso)

        return {"countgold":countgold,"countright":countright,"countpred":countpred,"inright":inright,
                "ingold":ingold,"ovright":ovright,"ovgold":ovgold},sens


    def get_score(self,cou:dict):
        countright=cou["countright"]
        countpred = cou["countpred"]
        countgold = cou["countgold"]
        ingold = cou["ingold"]
        inright = cou["inright"]
        ovgold = cou["ovgold"]
        ovright = cou["ovright"]

        def score(a,b):
            return (a)/(b+ 0.0)
        p = score(countright,countpred)
        r = score(countright,countgold)
        f1 = 2  * p * r / (p + r)
        ovrate = score(ovgold ,countgold) #+ 0.0)/ (countgold)
        inrate = score(ingold , countgold) #+ 0.0)/ (countgold)
        ovrecall = score(ovright,ovgold)
        inrecall = score(inright,ingold)
        return {"cous":cou,"p":p,"r":r,"f1":f1,"ovrate":ovrate,"inrate":inrate,"ovrecall":ovrecall,"inrecall":inrecall}







