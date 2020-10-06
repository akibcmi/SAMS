#-*-encoding=utf8-*-

import random
import codecs
import sys
import argparse

def opt():
    parser = argparse.ArgumentParser()
    #trainfiles
    parser.add_argument("--infile", type=str, default="")
    parser.add_argument("--count",type=int,default=2000)
    parser.add_argument("--name",type=str,required=True)
    parser.add_argument("--s",type=int,default=0,help="rand")
    return parser.parse_args()



if __name__ == "__main__":
    opts = opt()
    infiles = opts.infile
    name = opts.name
    count = opts.count

    trainfiles = "training/" + name + "_training.utf8"
    devfile = "training/" + name + "_dev.utf8"
    wordfile = "training/" + name + "_training_words.utf8"

    s = opts.s

    random.seed(s)

    files = codecs.open(infiles , "r" , "utf8")
    trainfile = codecs.open(trainfiles, "w","utf8")
    devfile = codecs.open(devfile, "w", "utf8")
    wordfiles = codecs.open(wordfile,"w","utf8")

    lines = files.readlines()
    linescous = len(lines)
    devlinesids = set()
    lastid = -1
    while len(devlinesids) < count:
        nextid = random.randint(0,linescous-1)
        if lastid == -1:
            lastid = nextid
        elif lastid == nextid:
            random.seed(lastid)
            continue
        devlinesids.add(nextid)
        lastid = nextid

    #for id in devlinesids:
    #    devfile.write(lines[id])

    for j in range(linescous):
        if j in devlinesids:
            devfile.write(lines[j])
        else:
            trainfile.write(lines[j])
            line = lines[j].strip("\n").strip().split()
            for word in line:
                wordfiles.write(word + "\n")



    files.close()
    trainfile.close()
    devfile.close()
    wordfiles.close()






