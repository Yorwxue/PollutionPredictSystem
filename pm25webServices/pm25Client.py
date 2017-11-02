import json
import requests
import PM25Config
import json

import requests

import PM25Config


def getHistoryData():
    config = PM25Config.getConfig()


    payload = {'hours': 12}

    jsonDataStr = requests.post("http://"+config['WEB_SERVICE_IP']+":"+str(config['WEB_SERVICE_PORT'])+"/getHistoryData",
                                json=json.dumps(payload))

    jsonData = json.loads(jsonDataStr.text)

    for siteName in jsonData:
        print(siteName)
        print(jsonData[siteName])


def insertPredictValue():
    config = PM25Config.getConfig()

    payload = {'左營': {'now_time': '2017-10-24 11:00', 'now_pm25': 20, '1hr': 25, '3hr': 30, '6hr': 40, '12hr': 60},
               '板橋': {'now_time': '2017-10-24 11:00', 'now_pm25': 25, '1hr': 21, '3hr': 50, '6hr': 10, '12hr': 20}}
    resStr = requests.post("http://"+config['WEB_SERVICE_IP']+":"+str(config['WEB_SERVICE_PORT'])+"/insertPredictValue",
                           json=json.dumps(payload))

    jsonData = json.loads(resStr.text)

    print(jsonData['result'])


getHistoryData()

# insertPredictValue()