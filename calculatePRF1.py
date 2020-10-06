#-*-encoding=utf8-*-


import codecs
import argparse

def createopt():
    arg = argparse.ArgumentParser()
    arg.add_argument("--pred",type=str,required=True)
    arg.add_argument("--gold",type=str,required=True)
    arg.add_argument("--word",type=str,required=True)
    return arg.parse_args()


if __name__ == "__main__":
    opt = createopt()

    with codecs.open(opt.pred , "r", "utf8") as p:
        with codecs.open(opt.gold,"r","utf8") as g:

            words = set()
            with codecs.open(opt.word,"r","utf8") as w:
                for line in w.readlines():
                    line = line.strip("\n").strip()
                    words.add(line)


            plines = p.readlines()
            glines = g.readlines()
            assert len(plines) == len(glines)

            predword=0
            predright=0
            goldword=0

            inword=0
            ovword=0

            inright=0
            ovright=0

            inrights=set()
            outrights=set()
            inwords=set()
            ovwords=set()
            goldwords=set()

            for j in range(len(plines)):
                pline = plines[j]
                gline = glines[j]

                senlen = len("".join(pline.strip("\n").strip().split()))

                pline = " ".join(pline.strip("\n").strip().split())
                gline = " ".join(gline.strip("\n").strip().split())

                k =0
                l =0
                c =0
                trueword = True
                pword = ""
                gword = ""
                lword = ""

                while c < senlen:
                    p = pline[k]
                    g = gline[l]
                    if p == " ":
                        predword = 1 + predword
                        k = k + 1
                        pword = pline[k]
                    else:
                        pword = pword + p
                    if g == " ":
                        goldword = 1 + goldword
                        goldwords.add(lword)
                        lword = gword
                        if lword in words:
                            inword = inword + 1
                            inwords.add(lword)
                        else:
                            ovword = ovword + 1
                            ovwords.add(lword)
                        l = l + 1
                        gword = gline[l]
                    else:
                        gword = gword + g

                    if p != g:
                        trueword = False
                    elif p == " ":
                        if trueword:
                            predright = predright + 1
                            if lword in words:
                                inright = inright + 1
                                inrights.add(lword)
                            else:
                                ovright = ovright + 1
                                outrights.add(lword)
                        trueword = True

                    k = k + 1
                    l = l + 1
                    c = c + 1
                predword = predword + 1
                goldword = goldword + 1
                lword = gword
                if lword in words:
                    inword = inword + 1
                    inwords.add(lword)
                else:
                    ovword = ovword + 1
                    ovwords.add(lword)
                if trueword:
                    predright = predright + 1
                    if lword in words:
                        inright = inright + 1
                    else:
                        ovright = ovright + 1


            def score(a, b):
                return (a) / (b + 0.0)


            p = score(predright, predword)
            r = score(predright, goldword)
            f1 = 2 * p * r / (p + r)
            ovrate = score(ovword, goldword)  # + 0.0)/ (countgold)
            inrate = score(inword, goldword)  # + 0.0)/ (countgold)
            ovrecall = score(ovright, ovword)
            inrecall = score(inright, inword)

            print("Right: {0}   Gold: {1}   All:{2} ".format(predright,goldword,predword))
            print("OVRight: {0}  OVGold: {1}  IVRight: {2} IVGold: {3}".format(ovright,ovword,inright,inword))
            print("P: {0}   \nR: {1}   \nF1: {2}   \nOOV Rate: {3}\nIV Rate: {4}\nOOV Recall: {5}\nIV Recall: {6}".format(p,r,f1,ovrate,inrate,ovrecall,inrecall))



