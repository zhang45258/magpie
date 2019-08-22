# coding:utf-8
import json
from wsgiref.simple_server import make_server
from xiangsidu import Xiangsidu,send
train_dict = Xiangsidu()
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
    mag1 = train_dict.calc_sim(text)  #首次匹配，结果是一个list，三维
    mag2 = send(txt=text, data=mag1, url="http://127.0.0.1:6089")  #发送数据到二次匹配，返回二次匹配的数据

    print(mag2)

    dic = {'no': no,
           'result00': mag2[0][1], 'result01': mag2[0][0], 'result02': mag2[0][3]+'(问题编号：'+mag2[0][0]+')',
           'result10': mag2[1][1], 'result11': mag2[1][0], 'result12': mag2[1][3],
           'result20': mag2[2][1], 'result21': mag2[2][0], 'result22': mag2[2][3],
           'result30': mag2[3][1], 'result31': mag2[3][0], 'result32': mag2[3][3]
            }
    return [json.dumps(dic).encode('utf8')]


if __name__ == "__main__":
    port = 6088
    httpd = make_server("0.0.0.0", port, application)
    print("serving http on port {0}...".format(str(port)))
    httpd.serve_forever()

