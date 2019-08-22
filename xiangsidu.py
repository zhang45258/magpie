#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 计算相似度

import random
import json
import xlrd
import jieba
import re
import requests
from gensim.models.word2vec import Word2Vec



class Xiangsidu(object):
    def __init__(self):
        self.model = Word2Vec.load('save/embeddings/best')
        self.vocab = set(self.model.wv.vocab.keys())
        # set() 函数创建一个无序不重复元素集,返回新的集合对象。
        # model.wv.vocab.keys()查看所有的词
    # 计算相似度
    def calc_sim(self, txt):
        self.a = result(self.vocab)
        txt = txt.replace('\r','').replace('\n','').strip() #去掉换行符
        txt = clean_text(txt) #清洗文本
        txt = seg_text(txt) #分词
        txt_list = clear_word_from_vocab(txt, self.vocab) #删除不在词汇表中的词
        for i in self.a:
            similarity = self.model.n_similarity(txt_list, i[2])
            '''
            解释上一句： =余弦相似性
            model.n_similarity (ws1 ws2):计算两组单词之间的余弦相似性。
                        参数: ws1 (str列表)-单词序列。
                              ws2 (str列表)-单词序列。
                        返回:  ws1和ws2之间的相似性。
                        返回类型:numpy.ndarray
            '''
            i.append(similarity) #把余弦值放到第4个值
            #print(similarity)
        def take4(elem):
            return elem[4]
        self.a.sort(key=take4, reverse=True) #按照第4个值降序排序，返回重新排序的列表
        return self.a[:30] #返回前30名

# 读取知识库表格,返回一个二维list，其中2列嵌套了一个list
def result(vocab):
    a = []
    data = xlrd.open_workbook('知识库.xlsx')
    sheetnames = data.sheet_names()  # 返回book中所有工作表的名字
    for sheetname in sheetnames:
        sheet = data.sheet_by_name(sheetname)  # 通过名称获取表格
        cols = sheet.col_values(0)  # 第一列内容
        length = len(cols)
        for i in range(length):
            No = sheetname + '_' + str(int(sheet.row_values(i)[0]))
            question = sheet.row_values(i)[1].replace("\n", "") #去掉换行符
            train_word_list = clean_text(question)  # 清洗文本
            train_word_list = seg_text(train_word_list)  # 分词
            question_list = clear_word_from_vocab(train_word_list, vocab)  # 删除不在词汇表中的词,返回一个list
            answer = sheet.row_values(i)[2]
            if question_list: #有可能分词/删除词以后，空了，这样这个问题就不再显示
                a.append([No, question, question_list, answer])
    return a


# 清洗文本
def clean_text(content):
    text = re.sub(r'[+——！，；／·。？、~@#￥%……&*“”《》：（）［］【】〔〕□]+', '', content)
    text = re.sub(r'[▲!"#$%&\'()*+,-./:;<=>\\?@[\\]^_`{|}~]+', '', text)
    text = re.sub('\d+', '', text)
    text = re.sub('\s+', '', text)
    return text

# 利用jieba包进行分词，去掉停词，返回分词后的文本,zhongwenTingcibiao.txt是从网上下载的停词表
def seg_text(text):
    stop = [line.strip() for line in open('data/zhongwenTingcibiao.txt', encoding='utf-8-sig').readlines()]
    text_seged = jieba.cut(text.strip())
    outstr = ''
    for word in text_seged:
        if word not in stop:
            outstr += word
            outstr += " "
    return outstr.strip()


# 清理不在word2vec词汇表中的词语，返回一个list
def clear_word_from_vocab(word_list, vocab):
    new_word_list = []
    for word in word_list.split():
        if word in vocab:
            new_word_list.append(word)
    return new_word_list


#post方式提交json数据，并接收返回数据
def send(txt, data, url):
    #先将list包装成json
    python2json = {}
    python2json["listData"] = data
    python2json["txt"] = txt
    data2json = json.dumps(python2json).encode('utf8')
    print(type(data2json))
    print(data2json)

    #发送
    headers = {'Content-Type': 'application/json'}
    request = requests.post(url=url, headers=headers, data=data2json)
    request = request.json()
    #解开json
    #json2python = json.loads(request).encode('utf8')
    response = request["listData"]
    return response

if __name__ =="__main__":
    train_dict =Xiangsidu()
    print(train_dict.calc_sim('积分兑换的车票是否可以办理改签？'))


