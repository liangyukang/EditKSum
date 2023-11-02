from sys import argv
import os.path as op
filepath=argv[1]

samples=0
with open(op.join(filepath,'iterations.txt'),'r+') as f1:
    lines=f1.readlines()
    samples=len(lines)
    sum=0
    for line in lines:
        sum+=int(line.strip())
    print("average iterations:%.1f" %(sum/samples))
    f1.write("average iterations:%.1f" %(sum/samples))

with open(op.join(filepath,'actions.txt'),'r+') as f2:
    lines=f2.readlines()
    print("average actions:%.1f" %(len(lines)/samples-1))
    f2.write("average actions:%.1f" %(len(lines)/samples-1))
