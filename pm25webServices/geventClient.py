import sys
import json
import requests

# conv = [{'input': 'hi', 'topic': 'Greeting'}]
# s = json.dumps(conv)
#
# print(s)




def getHistoryData():
    payload = {'hours': 12}

    jsonDataStr = requests.post("http://10.10.53.236:5000/getHistoryData", json=json.dumps(payload))

    jsonData = json.loads(jsonDataStr.text)

    for siteName in jsonData:
        print(siteName)
        print(jsonData['二林'])


def insertPredictValue():
    payload = {'二林': {'1hr': 25, '3hr': 30, '6hr': 40, '12hr': 60}, '板橋': {'1hr': 15, '3hr': 20, '6hr': 10, '12hr': 20}}
    resStr = requests.post("http://10.10.53.206:5000/insertPredictValue", json=json.dumps(payload))

    jsonData = json.loads(resStr.text)

    print(jsonData['result'])

# getHistoryData()

if __name__ == '__main__':
    getHistoryData()
    # insertPredictValue()
