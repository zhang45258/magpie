# -*- coding: utf-8 -*-
import os

file_dir = 'D:\\1\\1'
Ltxt = []
Llab = []
for root, dirs, files in os.walk(file_dir):
    print(root)
    for file in files:
        if os.path.splitext(file)[1] == '.txt':
            Ltxt.append(os.path.join(os.path.splitext(file)[0]))
        elif os.path.splitext(file)[1] == '.lab':
            Llab.append(os.path.join(os.path.splitext(file)[0]))
print(str(Ltxt))
print(str(Llab))

Ltxt2lab = []
Llab2txt = []
for item1 in Ltxt:
    if item1 not in Llab:
        Ltxt2lab.append(item1)

for item2 in Llab:
    if item2 not in Ltxt:
        Llab2txt.append(item2)
print('txt文件多出的'+str(Ltxt2lab))
print('lab文件多出的'+str(Llab2txt))

for txtfile in Ltxt2lab:
    os.remove(file_dir+'\\'+txtfile+'.txt')
for labfile in Llab2txt:
    os.remove(file_dir+'\\'+labfile+'.lab')
