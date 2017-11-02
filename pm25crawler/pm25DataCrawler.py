#!/home/kao/virtualenv/pm25/bin/python3

import urllib.request
from urllib.error import URLError, HTTPError
import schedule
import time
import ssl
import json
import logging
import pymysql
from logging import handlers
from datautil import pushNull
from socket import timeout


def logInit():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s'
    )

    tfh = handlers.TimedRotatingFileHandler('../log/pm25.log', when='midnight', backupCount=0, delay=False)

    fileformatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    tfh.setFormatter(fileformatter)
    tfh.suffix = '%Y-%m-%d'
    tfh.level = logging.ERROR

    logger = logging.getLogger('pm2.5')
    logger.addHandler(tfh)

    return logger


def dataProcess(logger, parameters):
    dataJson = getDataJson(logger)

    if dataJson is not None:

        lastestPublishTime = dataJson["result"]["records"][0]['PublishTime']
        # print(lastestPublishTime)

        if lastestPublishTime != parameters['last_update_time']:
            insertData(dataJson)
            parameters['last_update_time'] = lastestPublishTime


def insertData(dataJson):
    for i in dataJson["result"]["records"]:
        SiteName = i['SiteName']
        County = i['County']
        PSI = ''  # str(i['PSI'])
        MajorPollutant = str(i['Pollutant'])
        Status = str(i['Status'])
        SO2 = str(i['SO2'])
        CO = str(i['CO'])
        O3 = pushNull(SiteName, 'O3', str(i['O3']))
        PM10 = str(i['PM10'])
        PM2_5 = pushNull(SiteName, 'PM2_5', str(i['PM2.5']))
        NO2 = str(i['NO2'])
        WindSpeed = pushNull(SiteName, 'WindSpeed', str(i['WindSpeed']))
        WindDirec = pushNull(SiteName, 'WindDirec', str(i['WindDirec']))
        FPMI = ''  # str(i['FPMI'])
        NOx = str(i['NOx'])
        NO = str(i['NO'])
        PublishTime = str(i['PublishTime'])

        if PM2_5 == 'ND' or PM2_5 is None:
            PM2_5 = 'nan'

        sqlinsert = "INSERT INTO AirDataTable (SiteName, County, PSI, MajorPollutant, Status, SO2, CO, O3, PM10, PM2_5, NO2, WindSpeed, " \
                    "WindDirec, FPMI, NOx, NO, PublishTime) VALUES ('" + SiteName + "', '" + County + "', '" + PSI + "', '" + MajorPollutant + "', '" + Status + "', '" + SO2 + "', '" + CO + "', '" + O3 + "', '" + PM10 + "', '" + PM2_5 + "', '" + NO2 + "', '" + WindSpeed + "', '" + WindDirec + "', '" + FPMI + "', '" + NOx + "', '" + NO + "', '" + PublishTime + "')"
        try:

            conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='icrd00', charset='UTF8',
                                   db='AirData')
            cur = conn.cursor()  # 獲取一個游標對象
            cur.execute(sqlinsert)  # 插入數據
            cur.close()  # 關閉游標
            conn.commit()  # 向資料庫中提交任何未解決的事務，對不支持事務的資料庫不進行任何操作
            conn.close()  # 關閉到資料庫的連接，釋放資料庫資源
        except Exception as e:
            print(e)


def getDataJson(logger):

    context = ssl._create_unverified_context()

    try:
        fp = urllib.request.urlopen(
            "https://opendata.epa.gov.tw/webapi/api/rest/datastore/355000000I-000259?sort=SiteName&offset=0&limit=1000",
            context=context, timeout=10)
        mybytes = fp.read()
        mystr = mybytes.decode("utf8")
        fp.close()
        fjson = json.loads(mystr)

    except HTTPError as e:
        logger.error('HTTP Error code: ' + str(e.code))
        logger.error('HTTP Error: ' + e.reason)
    except URLError as e:
        logger.error('URL error: ' + e.reason)
    except timeout:
        logger.error('url timed out...')
    else:
        return fjson


if __name__ == "__main__":


    parameters = {}
    parameters['last_update_time'] = ''

    logger = logInit()

    schedule.every().minute.do(dataProcess, logger, parameters)

    while True:
        schedule.run_pending()
        time.sleep(1)











