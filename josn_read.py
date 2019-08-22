import json

'''
用于将json文件转为txt文件。用于转换通用语料库
'''

n = 0
d_one = []
j = open('C:\\data\\nlp_chinese_download\\baike_qa_train.json',  encoding='utf-8')  #设置以utf-8解码模式读取文件，encoding参数必须设置，否则默认以gbk模式读取文件，当文件中包含中文时，会报错
for line in j:
    test_dict = json.loads(line)
    d_one.append(test_dict)
    print(test_dict)
    #每1000行更换一个文件名
    m = n//1000
    #写入
    with open('data/nlp_chinese_corpus/train'+str(m)+'.txt', 'a', encoding='utf-8') as f:
        f.write(str(test_dict.get('answer')))
    n = n+1
