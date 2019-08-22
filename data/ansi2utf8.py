# -*- coding: utf-8 -*-
import os

file_dir = 'D:\\1\\1'

#全部转为utf8格式
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            L.append(os.path.join(root, file))
    return L

for file in file_name(file_dir):
    try:
        bg = open(file, 'r')
        ut = open(os.path.splitext(file)[0]+'utf8'+os.path.splitext(file)[1], 'w', encoding='utf-8')
        ut.write(bg.read())
        bg.close()
        ut.close()
        os.remove(file)
    except:
        print(file + '  请检查文件格式!!!')
    continue