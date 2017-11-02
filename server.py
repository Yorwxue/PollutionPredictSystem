# _*_ coding: utf-8 _*_

from __future__ import print_function

import gc
import json
import os
import subprocess
import sys
# import zmq
# import threading
import time
from datetime import datetime

import requests

import PM25Config
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
from config import get_config, site_map

config = get_config()
root_path = config['root_path']

# url_slave = 'inproc://ping-workers'
# url_router = 'tcp://10.10.53.210:9527'

# gauth = GoogleAuth()
# gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication.
# drive = GoogleDrive(gauth)

# server_ip = 'http://10.10.53.238:5000'
payload = {'hours': config['period']}

training_year = '2014-2016'  # 2014-2015
testing_year = '2017-2017'
training_duration = '1/1-12/31'  # '1/1-11/30'
testing_duration = '1/1-1/31'
target_kind = 'PM2.5'
pollution_kind = ['PM2.5', 'O3', 'WIND_SPEED', 'WIND_DIREC']  # , 'AMB_TEMP', 'RH'
# interval_hours = 6  # predict the label of average data of many hours later, default is 1
is_training = '0'  # True: 1; False: 0

pollution_site_map = site_map()


def time_spent_printer(start_time, final_time):
    spent_time = final_time - start_time
    print('totally spent ', end='')
    print(int(spent_time / 3600), 'hours ', end='')
    print(int((int(spent_time) % 3600) / 60), 'minutes ', end='')
    print((int(spent_time) % 3600) % 60, 'seconds')


def server_slave(name, payload, url_slave, context):
    config = PM25Config.getConfig()

    # slave = context.socket(zmq.REP)
    # slave.connect(url_slave)
    # while True:
    print('slave {0} start'.format(name))
    # message = slave.recv_json()
    jsonDataStr = requests.post("http://"+config['WEB_SERVICE_IP']+":"+str(config['WEB_SERVICE_PORT'])+"/getHistoryData",
                                json=json.dumps(payload))
    message = json.loads(jsonDataStr.text)

    print('Data Length: %d' % len(message))
    print('slave {0} recv "{1}"'.format(name, message))
    gc.disable()
    start_time = time.time()

    with open('ensemble_regression/slave_%d_msg' % name, 'w') as fw:
        json.dump(message, fw)

    # ------------------------------------------------------------------------------

    def paralPredict(local, city, target_site, pollution_kind, message, training_year, testing_year,
                       training_duration, testing_duration, interval_hours, is_training, result):
        # data = reader.data_collection(local, city, pollution_kind, message)
        if not(target_site in result):
            result[target_site] = dict()
        # print(sorted(message.keys())[-1])
        args = ' '.join([target_kind, local, city, target_site, interval_hours, is_training, str(name)])

        program = 'python3.6 ensemble_regression/ensemble_model.py'
        try:
            process = subprocess.Popen(program + ' ' + args, shell=True, stdout=subprocess.PIPE)

            value, err = process.communicate()  # 主要用來取回 stdout 跟 stderr 的輸出

            # print('---')
            # print(value.split()[-1])
            result[target_site][interval_hours] = float(value.split()[-1])
            print('site: %s %d hour result = %f' % (target_site, int(interval_hours), result[target_site][interval_hours]))
        except:
            # raise
            result[target_site][interval_hours] = 'NaN'
            print('site: %s %d hour result = NaN' % (target_site, int(interval_hours)))
            tp, val, tb = sys.exc_info()
            with open('errmsg', 'a') as f:
                f.write('%s: %s %s\n' % (str(sorted(message.keys())[-1]), str(tp), str(val)))
                # print >> sys.stderr, '%s: %s' % (str(tp), str(val)
        # gc.enable()

    # ------------------------ 1hr ----------------------------
    result = dict()
    interval = '1'
    for location in sorted(pollution_site_map.keys()):
        for city in sorted(pollution_site_map[location].keys()):
            for site in sorted(pollution_site_map[location][city]):
                paralPredict(location, city, site, pollution_kind, message, training_year, testing_year, training_duration,
                             testing_duration, interval, is_training, result)
                break
            break
        break

    payload = result
    resStr = requests.post(
        os.path.join("http://", config['WEB_SERVICE_IP'] + ":" + str(config['WEB_SERVICE_PORT']), "insertPredictValue_1hr"),
        json=json.dumps(payload))
    del result

    # ------------------------ 6hr ----------------------------
    result = dict()
    interval = '6'
    for location in sorted(pollution_site_map.keys()):
        for city in sorted(pollution_site_map[location].keys()):
            for site in sorted(pollution_site_map[location][city]):
                paralPredict(location, city, site, pollution_kind, message, training_year, testing_year,
                             training_duration,
                             testing_duration, interval, is_training, result)
                break
            break
        break

    payload = result
    resStr = requests.post(
        os.path.join("http://", config['WEB_SERVICE_IP'] + ":" + str(config['WEB_SERVICE_PORT']), "insertPredictValue_6hr"),
        json=json.dumps(payload))
    del result

    # ------------------------ 12hr ----------------------------
    result = dict()
    interval = '12'
    for location in sorted(pollution_site_map.keys()):
        for city in sorted(pollution_site_map[location].keys()):
            for site in sorted(pollution_site_map[location][city]):
                paralPredict(location, city, site, pollution_kind, message, training_year, testing_year,
                             training_duration,
                             testing_duration, interval, is_training, result)
                break
            break
        break

    payload = result
    resStr = requests.post(
        os.path.join("http://", config['WEB_SERVICE_IP'] + ":" + str(config['WEB_SERVICE_PORT']), "insertPredictValue_12hr"),
        json=json.dumps(payload))
    del result
    # -----------------------------------------------------------------------------

    final_time = time.time()
    time_spent_printer(start_time, final_time)

    # msg = json.dumps(result)
    # slave.send_json(msg)
    # print('return status: %s' % str(json.loads(resStr)))
    print('finish process: slave {0}'.format(name))
    # del message
    gc.collect()
    gc.enable()

    # slave.close()
#
# slave_num = 3
#
# context = zmq.Context()
#
# router = context.socket(zmq.XREP)  # or zmq.ROUTER, their value are both 6
# router.bind(url_router)
#
# slaves = context.socket(zmq.XREQ)  # or zmq.DEALER, their value are both 5
# slaves.bind(url_slave)

# thread_pool = [i for i in range(slave_num)]
# for i in range(slave_num):
#     # thread = threading.Thread(target=server_slave, args=(i, url_slave, context,))
#     # thread.start()
#     thread_pool[i] = threading.Thread(target=server_slave, args=(i, url_slave, context,))
#     thread_pool[i].start()
#
# zmq.device(zmq.QUEUE, router, slaves)

# while True:
#     for i in range(slave_num):
#         if not thread_pool[i].isAlive():
#             thread_pool[i] = threading.Thread(target=server_slave, args=(i, url_slave, context,))
#             thread_pool[i].start()

# router.close()
# slaves.close()
# context.term()


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
    server_slave(9527, payload, None, None)

import schedule

if __name__ == '__main__':
    # while True:
    #     try:
    #         server_slave(9527, payload, None, None)
    #     except:
    #         time.sleep(600)  # ten minute
    #         server_slave(9527, payload, None, None)

    # logger = logInit()

    schedule.every().minute.do(checkDataUpdate)

    schedule.run_all()

    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)
