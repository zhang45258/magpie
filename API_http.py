# coding:utf-8
#多标签分类
import json
from wsgiref.simple_server import make_server
from magpie import Magpie
from xlsx_read import result

magpie1 = Magpie(
        keras_model='save/model/best.h5',
        word2vec_model='save/embeddings/best',
        scaler='save/scaler/best',
        labels=['1111', '1112', '1113', '1114', '1115', '1116', '1117', '1118', '1121', '1122', '1123', '1124', '1131',
                '1132', '1133', '1134', '1135', '1141', '1142', '1143', '1144', '1151', '1152', '1153', '1154', '1211',
                '1212', '1213', '1214', '1215', '1216', '1217', '1218', '1219', '1221', '1222', '1223', '1231', '1232',
                '1233', '1234', '1235', '1241', '1242', '1243', '1251', '1311', '1312', '1313', '1314', '1321', '1322',
                '1323', '1331', '1332', '1333', '1334', '1341', '1342', '1343', '1344', '1345', '1351', '1411', '1421',
                '1431', '1441', '15', '2111', '2112', '2113', '2114', '2115', '2116', '2117', '2121', '2122', '2123',
                '2124', '2131', '2132', '2133', '2134', '2141', '2142', '2143', '2144', '2145', '2146', '2147', '2148',
                '2149', '21410', '2151', '2152', '2153', '2154', '2155', '2156', '2161', '2162', '2163', '2164', '2165',
                '2166', '2167', '2168', '2171', '2172', '2173', '2174', '2175', '2176', '2177', '2178', '2179', '21710',
                '21711', '2181', '2182', '2183', '2184', '2185', '2186', '2187', '2188', '2191', '2192', '2193', '2194',
                '2195', '2196', '221', '222', '223', '224', '2311', '2312', '2313', '2314', '2315', '2316', '2321',
                '2322', '2323', '2324', '24', '31', '32', '33', '34', '41', '42', '43', '51', '52', '53', '54', '55',
                '56', '57', '58', '61', '7111', '7112', '7113', '7114', '7115', '7116', '7117', '7118', '7119', '71110',
                '71111', '7121', '7122', '7123', '7124', '7125', '7126', '7127', '7128', '7129', '7131', '7132', '7133',
                '7134', '7135', '7136', '7137', '7138', '7139', '71310', '71311', '71312', '7141', '7142', '7151',
                '721', '722', '723', '724', '7311', '7312', '7313', '7314', '7315', '7316', '7321', '7322', '7323',
                '7324', '7325', '7326', '7331', '7332', '7333', '7334', '7335', '7336', '734', '74']
                )

# 定义函数，参数是函数的两个参数，都是python本身定义的，默认就行了。
def application(environ, start_response):
    # 定义文件请求的类型和当前请求成功的code
    start_response('200 OK', [('Content-Type', 'application/json')])
    # environ是当前请求的所有数据，包括Header和URL，body
    request_body = environ["wsgi.input"].read(int(environ.get("CONTENT_LENGTH", 0)))
    request_body = json.loads(request_body.decode('utf-8'))
    text = request_body["text"]
    no = request_body["no"]
    print(text)
    if no == '1':
        mag = magpie1.predict_from_text(text)

    print(mag)
    key, value = zip(*mag)
    print(key)

    dic = {'no': no,
           'result00': result(key[0])[2], 'result01': result(key[0])[3],
           'result10': result(key[1])[2], 'result11': result(key[1])[3],
           'result20': result(key[2])[2], 'result21': result(key[2])[3],
           'result30': result(key[3])[2], 'result31': result(key[3])[3]}

    return [json.dumps(dic).encode('utf8')]


if __name__ == "__main__":
    port = 6088
    httpd = make_server("0.0.0.0", port, application)
    print("serving http on port {0}...".format(str(port)))
    httpd.serve_forever()

