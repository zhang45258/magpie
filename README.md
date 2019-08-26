## 客服中心智能问答系统首次匹配服务组件（包括模型生成）  
    本代码是论文开源代码的一部分。本代码的执行环境：win64， python3.x    
   ### 本代码用于：  
 ### 一、准备语料，训练生成 word2vec。  
    操作步骤：  
      1.将训练语料放置在'./data/hep-categories/'文件夹中。语料为.txt格式，utf-8编码。  
      2.执行train.py,生成model并保存。（保存地址：'./save/embeddings/best'）  
### 二、生成http接口，接收前台发送数据，返回首次匹配的数据  
    操作步骤：  
      1.执行Api_http.py（该程序执行以下操作）  
        (1)生成http接口。接收内容：post方式，json数据，{"txt":"客户问话原文"，"listData":"首次匹配结果list"}，返回内容：  
        json数据，{"listData":"二次匹配结果list"}。  
        （2）接收首次匹配服务组件发送json数据，解析json，将内容按规定格式写入到"./data_http.tsv"中。  
        （3）读取"./data_http.tsv"文档，使用model进行语义相似度计算。  
        （4）计算结果进行排序，截取前4名  
        （5）序列化成json数据并返回。
