from flask import Flask, Response
from flask import request

from flask_cors import CORS, cross_origin

from gevent import monkey
import gevent.wsgi

import werkzeug.serving

import json

import time


# monkey.patch_all()        #非同步處理，無法與debug模式一起使用

app = Flask(__name__)
CORS(app)

@app.route("/")
def helloWorld():
    # time.sleep(15)
    return "Hello, gevent cross-origin-world! \n"


@app.route("/play", methods=['GET', 'POST'])
def play():
    jsondata = request.get_json()

    print(jsondata)

    data = json.loads(jsondata)

    data['result'] = 'good'

    # time.sleep(5)

    return json.dumps(data)

@app.route("/play2")
def play2():
    # time.sleep(15)
    return "Hello, play2! \n"



@werkzeug.serving.run_with_reloader
def runServer():

    # use gevent WSGI server instead of the Flask

    app.debug = True    #無法與非同步一起使用

    ws = gevent.wsgi.WSGIServer(('', 5000), app.wsgi_app)
    ws.serve_forever()

if __name__ == '__main__':
    "Start gevent WSGI server"
    runServer()


