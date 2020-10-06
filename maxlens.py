# -*-encoding=utf8-*-

import sys
import codecs

files = sys.argv[1]

maxline=0
maxline2=0
with codecs.open(files , "r" , "utf8") as f1:
	for line in f1.readlines():
		if len(line.strip("\n").strip().split()) > maxline:
			maxline = len(line.strip("\n").strip().split())
		if len("".join(line.strip("\n").strip().split())) > maxline2:
			maxline2 = len("".join(line.strip("\n").strip().split()))	
print(maxline)
print(maxline2)
