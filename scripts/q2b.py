# -*- encoding=utf8 -*-
import codecs
import sys

files = sys.argv[1]
files2 = sys.argv[2]


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


with codecs.open(files , "r" , "utf8") as fs1:
    with codecs.open(files2 , "w" , "utf8") as f2:
        for line in fs1.readlines():
            f2.write(strQ2B(line))
