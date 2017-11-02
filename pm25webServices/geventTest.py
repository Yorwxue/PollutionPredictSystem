from flask import Flask, Response

from gevent.pywsgi import WSGIServer
from gevent import monkey

from flask_cors import CORS, cross_origin
import time


monkey.patch_all()

app = Flask(__name__)
CORS(app)

@app.route("/")
def helloWorld():
    # time.sleep(15)
    return "Hello, gevent cross-origin-world!\n"

@app.route("/play")
def play():
    # time.sleep(15)
    return "play!\n"

if __name__ == '__main__':
    # app.run(host='10.10.53.202', port=5000, debug=True)

    "Start gevent WSGI server"
    # use gevent WSGI server instead of the Flask
    http = WSGIServer(('', 5000), app.wsgi_app)
    # TODO gracefully handle shutdown
    http.serve_forever()
