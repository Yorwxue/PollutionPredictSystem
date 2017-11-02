from flask import Flask, Response
from flask import request

from flask_cors import CORS, cross_origin

from gevent import monkey
import gevent.wsgi

import werkzeug.serving

import json

from datetime import datetime, timedelta
import pymysql


# monkey.patch_all()        #非同步處理，無法與debug模式一起使用

app = Flask(__name__)
CORS(app)


@app.route("/getHistoryData", methods=['POST'])
def getHistoryData():
    jsondata = request.get_json()
    data = json.loads(jsondata)

    # print(data)

    hours = data['hours']

    now = datetime.now()
    startTime = (now - timedelta(hours=hours)).strftime("%y-%m-%d %H:%M")


    conn= pymysql.connect(host='localhost', port=3306, user='root', passwd='icrd00', charset='UTF8', db='AirData')
    cur = conn.cursor(pymysql.cursors.DictCursor)

    sqlStr = "select * from AirDataTable where PublishTime >= '" + startTime+"' order by PublishTime ASC"

    cur.execute(sqlStr)


    twelveHoursData = {}

    for row in cur:
        country = row['County']
        siteName = row['SiteName']

        if siteName in twelveHoursData:
            twelveHoursData[siteName].append(row)
        else:
            twelveHoursData[siteName] = []
            twelveHoursData[siteName].append(row)

    return json.dumps(twelveHoursData, ensure_ascii=False)



@app.route("/insertPredictValue", methods=['POST'])
def insertPredictValue():
    jsondata = request.get_json()
    data = json.loads(jsondata)

    for siteName in data:
        predict_1hr_value = data[siteName]['1hr']
        predict_3hr_value = data[siteName]['3hr']
        predict_6hr_value = data[siteName]['6hr']
        predict_12hr_value = data[siteName]['12hr']


        print(str(predict_1hr_value)+'\t'+str(predict_3hr_value)+'\t'+str(predict_6hr_value)+'\t'+str(predict_12hr_value))

        res = {}

        res['result'] = 'ok'


    return json.dumps(res, ensure_ascii=False)


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


