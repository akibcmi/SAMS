# -*- coding: UTF-8 -*-
import re
import argparse
import sys
import codecs

Maximum_Word_Length = 4

#def OT(str):
#    print str.encode('utf8')

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        
        rstring += chr(inside_code)
    return rstring

def preprocess(path):
    numslines = []
    engslines = []
    rNUM = u'((-|\+)?\d+((\.|·)\d+)?%?)'
    rENG = u'[A-Za-z_.]+'
    findNum=re.compile(rNUM)
    findEng=re.compile(rENG)
    #word_count, char_count, sent_count = 0, 0, 0
    #count_longws = 0
    with codecs.open(path,'r','utf8') as f:
        sents = []
        for line in f.readlines():
            sent = strQ2B(line).strip().strip('\n').split()
            new_sent = []
            numsline = []
            engsline = []
            for word in sent:
                word = re.sub(u'\s+','',word,flags =re.U)



                nums = findNum.findall(word)  # re.findall(rNUM,word,flags=re.U)
                nums = [num[0] for num in nums]
                numsline = numsline + nums
                word = re.sub(rNUM, u'0', word, flags=re.U)


                engs = findEng.findall(word)
                engsline = engsline + engs #line+engs
                word = re.sub(rENG, u'X', word)

                new_sent.append(word)

            numslines.append(numsline)
            engslines.append(engsline)
            sents.append(new_sent)
    #print  path
    #print 'long words count', count_longws
    #print  'sents %d, words %d chars %d' %(sent_count, word_count, char_count)
    return sents,numslines,engslines

def getnums(line):
    line = re.sub(u'\s+',"",line,flags=re.U).strip("]").strip("[").strip()
    if line == "":
        return []
    line=line.split(',')
    line = [s for s in line]
    return line

def getengs(line):
    line = re.sub(u'\s+', ""  , line, flags=re.U).strip("]").strip("[").strip()
    if line == "":
        return []
    line=line.split(',')
    #if line.strip == ''
    #    .split(',')
    line = [s for s in line]
    return line

def reps(path,pathnums,pathengs):

    fs=codecs.open(path,'r','utf8')
    fnums=codecs.open(pathnums,'r','utf8')
    fengs=codecs.open(pathengs,'r','utf8')
    #fslines=fs.readlines()
    #fnumslines = fnums.readlines()
    #fengslines=fengs.readlines()
    sens=[]
    for lines,numslines,engslines in zip(fs.readlines(),fnums.readlines(),fengs.readlines()):
        new_sens = []
        lines = lines.strip("\n").strip().split()
        numslines = getnums(numslines)
        engslines=getengs(engslines)
        numsj=0
        engsj=0
        if len(engslines) == 0 and len(numslines) == 0:
            sens.append(lines)
            continue

        for line in lines:
            #print(line)
            lastx=0
            last0=0
            linex=""
            for cha in line:
                if cha == u"X":
                    linex += engslines[engsj]
                    engsj = 1 +engsj
                else:
                    linex=linex+cha
            line = linex
            linex=""
            for cha in line:
                if cha == u'0':
                    linex =linex + numslines[numsj]
                    numsj =numsj + 1
                else:
                    linex=linex+cha
            #while u"X" in line:
            #    line=line.replace(u"X",engslines[engsj],1)
            #    engsj = 1+engsj
            #while u'0' in line:
            #    line=line.replace(u'0',numslines[numsj],1)
            #    numsj=numsj+1
            #print(line)
            new_sens.append(linex)
        sens.append(new_sens)
        assert numsj == len(numslines)
        assert len(engslines)==engsj
    return sens


def replacespace(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        rstring += chr(inside_code)
    return rstring
def B2Q(path,sens):

    with codecs.open(path,'r','utf8')as f:
        print(path)
        lines=f.readlines()
#        print(len(lines))
        print(len(sens))
        assert len(lines)==len(sens)
        line3=[]
        for line , ses1  in zip(lines,sens):
            line=replacespace(line)
            line=re.sub(u"\s+","",line,flags=re.U)
            line2=[]
            j = 0
            for ses2 in ses1:
                ses2 = re.sub(u'\s+',"",ses2,flags=re.U)
                ses3 = []
                for cha in ses2:
                    if cha == line[j] or ord(cha) + 65248 == ord(line[j]):
                        ses3.append(line[j])
                        j = 1 + j
                    else:
                        assert False
                line2.append("".join(ses3))
            line3.append(line2)
        return line3
def replaceFiles(path,sens):
    with codecs.open(path,'r','utf8')as f:
        print(path)
        sens2 = []
        for line, ses1 in zip (f.readlines() , sens):
            line = line.strip("\n").strip()
            #print(line)
            #print("".join(ses1))
            assert(len("".join(line.split())) == len("".join(ses1)))
            line2 = ""
            for uch in line:
                inside_code = ord(uch)
                if inside_code == 12288:
                    inside_code = 32
                line2 = line2 + chr(inside_code)
            line2 = re.sub(u"\s+","",line2,flags=re.U)
            ses2 = []
            j=0
            for ses12 in ses1:
                ses12 = re.sub(u'\s+',"",ses12,flags=re.U)
                cous = j
                lcs = 0
                line3 = ""
                while lcs < len(ses12):
                    line3 = line3 + line2[cous]
                    if line2[cous] != ' ':
                        lcs = lcs + 1
                    cous = cous + 1 
                ses2.append(line3) #.substring(j,len(ses12)))
                j = cous
            sens2.append(ses2)
        return sens2
                
def lines(dataset1,dataset2):
    with codecs.open(dataset1,"r","utf8")as f1:
        with codecs.open(dataset2,"w","utf8")as f2:
            for line in f1.readlines():
                line = line.strip("\n").strip()
                sens = ""
                biao = "。？?！!;；“\":："
                for j in range(len(line)):
                    sens = sens + line[j]
                    if line[j] in biao:
                        f2.write(sens + "\n")
                        sens = ""
                if len(sens) > 0:
                    f2.write(sens)
                    f2.write("\n")
                #line = re.split(r"([。？?！!])",line.strip("\n").strip())
                #for lines in line:
                #    f2.write(lines + "\n")
def lines2(dataset,sens):
    with codecs.open(dataset,"r","utf8")as f1:
        j = 0
        senalls = []
        for lines in f1.readlines():
            lines = "".join(lines.strip("\n").strip().split())
            lengths = 0
            sen = []
            while lengths != len(lines):
                sen1 = []
                for s in sens[j]:
                    sen1 .append( s)
                lengths = lengths + len("".join(sen1))
                sen = sen + sens[j]
                j = j     + 1
            senalls.append(sen)
        return senalls

def check(dataset1,dataset2):
    with codecs.open(dataset1,"r","utf8")as f1:
        with codecs.open(dataset2, "r", "utf8")as f2:
            for line1,line2 in zip(f1.readlines(),f2.readlines()):
                line1 = line1.strip("\n").strip().split()
                line2 = line2.strip("\n").strip().split()
                line1 = "".join(line1)
                line2 = "".join(line2)
                assert  line1 == line2
def writesdatasets(datas,paths):
    with codecs.open(paths,'w','utf8')as f:
        for line in datas:
            f.write(" ".join(line) + "\n")
def writea(datas,paths):
    with codecs.open(paths,'w','utf8')as f:
        for line in datas:
            f.write("["+",".join(line) + "]")
            f.write("\n")
def create_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument("--type",default='produces' , choices=['repos','produces'])
    parser.add_argument("--dataset",type=str,required=True)
    parser.add_argument("--to",type=str,required=True)
    parser.add_argument("--engs",type=str,required=True)
    parser.add_argument("--nums",type=str,required=True)
    parser.add_argument("--lines",type=str,default=None)
    parser.add_argument("--dataset2",type=str,default=None)
    parser.add_argument("--replace" , default="", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    arg = create_opts()
    type=arg.type #sys.argv[1]
    dataset=arg.dataset #sys.argv[2]
    todataset=arg.to #sys.argv[3]
    numdataset=arg.nums #sys.argv[4]
    engsdataset=arg.engs #sys.argv[5]
    linesdataset=arg.lines #sys.argv[6]
    print(todataset)
    if type == 'repos':
        dataset2 = arg.dataset2 #sys.argv[7]
    if type =='produces' :
        if linesdataset is None:
            todata,numdatas,engsdatas=preprocess(dataset)
        else:
            lines(dataset,linesdataset)
            todata,numdatas,engsdatas=preprocess(linesdataset)
        writesdatasets(todata,todataset)
        writea(numdatas,numdataset)
        writea(engsdatas,engsdataset)
    else:
        sens=reps(todataset,numdataset,engsdataset)
        if linesdataset is None:
            sens=B2Q(dataset,sens)
        else:
            sens=B2Q(linesdataset,sens)
            sens=lines2(dataset , sens)
        if arg.replace !=  "":
            sens = replaceFiles(arg.replace , sens) 
        #if linesdataset is "None":
        writesdatasets(sens,dataset2)
        if arg.replace ==  "":
            check(dataset,dataset2)
        else:
            check(arg.replace,dataset2)
