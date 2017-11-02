import gevent.wsgi
import werkzeug.serving

from flask import Flask, Response
from flask_cors import CORS, cross_origin

from gevent import monkey


# monkey.patch_all()

app = Flask(__name__)
CORS(app)

@app.route("/")
def helloWorld():
    # time.sleep(5)
    return "Hello, gevent cross-origin-world! \n"

@werkzeug.serving.run_with_reloader
def runServer():
    app.debug = True

    ws = gevent.wsgi.WSGIServer(('', 5000), app)
    ws.serve_forever()

runServer()