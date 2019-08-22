# -*- coding: utf-8 -*-
import codecs
import os
from win32com import client as wc
'''
将doc/docx文件转为txt文件。
用于转换铁路规章文电，转换后将原doc/docx文件删除。
有时文件有问题，会无法转换。需要关闭程序，手工从任务管理器中关闭word进程，然后将该文件删除，继续运行程序
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


#将doc文件另存为txt文件
def doc2txt(file_dir):
    word = wc.Dispatch('Word.Application')
    for L in file_name(file_dir, '.doc'):
        print(L)
        doc = word.Documents.Open(L)
        doc.SaveAs(os.path.splitext(L)[0]+'.txt', 2)
        doc.Close()
        os.remove(L)
    word.Quit()

#将docx文件另存为txt文件
def docx2txt(file_dir):
    word = wc.Dispatch('Word.Application')
    for L in file_name(file_dir, '.docx'):
        print(L)
        doc = word.Documents.Open(L)
        doc.SaveAs(os.path.splitext(L)[0]+'.txt', 2)
        doc.Close()
        os.remove(L)
    word.Quit()

#将ANSI编码的txt文件转为utf-8编码
def ansi2utf8(file_dir):
    for L in file_name(file_dir, '.txt'):
        try:
            bg = open(L, 'r')
            ut = open(os.path.splitext(L)[0]+'utf8'+'.txt', 'w', encoding='utf-8')
            ut.write(bg.read())
            bg.close()
            ut.close()
            os.remove(L)
        except:
            print(L + '  请检查文件格式!!!')
        continue


#doc2txt('C:\\data\\Railway_Passenger_Transport_download')
#docx2txt('C:\\data\\Railway_Passenger_Transport_download')
ansi2utf8('C:\\data\\Railway_Passenger_Transport')

