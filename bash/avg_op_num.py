import sys
import numpy as np
srcs=open(sys.argv[1],'r').readlines()
tgt=open(sys.argv[2],'w')
rep=[]
dele=[]
ins=[]
for i,src in enumerate(srcs):
    
    ops=src.strip()[1:-1]

    ops=ops.split(', ')
    rep.append(float(ops[0]))
    dele.append(float(ops[1]))
    ins.append(float(ops[2]))
avg_rep=np.mean(rep)
avg_del=np.mean(dele)
avg_ins=np.mean(ins)
tgt.write('%.1f %.1f %.1f\n' % (avg_rep,avg_del,avg_ins))
