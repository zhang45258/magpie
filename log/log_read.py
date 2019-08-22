# -*- coding: utf-8 -*-
import codecs
import os
import json
import re
'''
读取log文件夹内的txt文件，提取val_loss的列表，选取其中最小值。
然后比较，得到所有log文件的最小参数

'''

#获取目录下的所有文件名
def file_name(file_dir, file_suffix):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == file_suffix:
                L.append(os.path.join(root, file))
    print(L)
    return L

dir = 'filedir'
Epochs = 0
value = 1.0

for L in file_name('C:\\magpie-master\\log' , '.txt'):
    with open(L, 'r') as file_open:
        data = file_open.read()
        data = re.sub('\'','\"',data)
        data = json.loads(data)
        a = data['val_loss']
        print(a)  #val_loss值的列表
        #对列表进行排序
        b = list(zip(a, range(len(a))))
        b.sort(key=lambda x: x[0])
        c = [x[1] for x in b]
        print(c)  #val_loss值的列表升序排序，得到下标列表
        #把最小的值组合成一个列表
        if value >= a[c[0]]:
            dir = L
            Epochs = c[0]
            value = a[c[0]]
print('dir=', dir)
print('Epochs=', Epochs)
print('value=', value)