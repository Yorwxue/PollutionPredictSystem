
import schedule
import logging
from logging import handlers
import time
import requests
import json
from datetime import datetime
import PM25Config



# def logInit():
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s %(levelname)-8s %(message)s'
#     )
#
#     tfh = handlers.TimedRotatingFileHandler('../log/pm25.log', when='midnight', backupCount=0, delay=False)
#
#     fileformatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
#     tfh.setFormatter(fileformatter)
#     tfh.suffix = '%Y-%m-%d'
#     tfh.level = logging.ERROR
#
#     logger = logging.getLogger('pm2.5')
#     logger.addHandler(tfh)
#
#     return logger



def checkDataUpdate():

    config = PM25Config.getConfig()

    jsonDataStr = requests.get("http://"+config['WEB_SERVICE_IP']+":"+str(config['WEB_SERVICE_PORT'])+"/getPredictStatus")

    jsonData = json.loads(jsonDataStr.text)

    dataUpdateTime = jsonData['dataUpdateTime']
    predictionTime = jsonData['predictionTime']



    if dataUpdateTime != predictionTime:
        nowTimeStr = datetime.now().strftime("%Y-%m-%d %H:00")

        payload = {}
        payload['nowTime'] = nowTimeStr

        requests.post("http://"+config['WEB_SERVICE_IP']+":"+str(config['WEB_SERVICE_PORT'])+"/insertPredictStatus",
                      json=json.dumps(payload))

        doPredict()


def doPredict():
    print('doing predict')


if __name__ == "__main__":


    # logger = logInit()

    schedule.every().minute.do(checkDataUpdate)

    schedule.run_all()

    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)

