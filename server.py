# _*_ coding: utf-8 _*_

from __future__ import print_function
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import zmq
import threading
import numpy as np
import time
import json

# import rnn, reader
import ensemble_model, reader

# target_program = 'test2.py'
url_slave = 'inproc://ping-workers'
# url_router = 'tcp://127.0.0.1:9527'
url_router = 'tcp://10.10.53.205:9527'

gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)


def time_spent_printer(start_time, final_time):
    spent_time = final_time - start_time
    print('totally spent ', end='')
    print(int(spent_time / 3600), 'hours ', end='')
    print(int((int(spent_time) % 3600) / 60), 'minutes ', end='')
    print((int(spent_time) % 3600) % 60, 'seconds')


def server_slave(name, url_slave, context):
    print('slave {0} start'.format(name))
    slave = context.socket(zmq.REP)
    slave.connect(url_slave)
    while True:
        result = dict()
        try:
            message = slave.recv_json()
            print('slave {0} recv "{1}"'.format(name, message))
            start_time = time.time()

            training_year = '2014-2015'  # 2014-2015
            testing_year = '2015-2015'
            training_duration = '1/1-11/30'  # '1/1-11/30'
            testing_duration = '12/1-12/31'
            target_kind = 'PM2.5'
            pollution_kind = ['PM2.5', 'O3', 'AMB_TEMP', 'RH', 'WIND_SPEED', 'WIND_DIREC']
            # interval_hours = 6  # predict the label of average data of many hours later, default is 1
            is_training = False

            # ------------------------------------------------------------------------------

            def paralPredict(local, city, target_site, target_kind, message, training_year, testing_year,
                               training_duration, testing_duration, interval_hours, is_training, result):
                data = reader.data_collection(local, city, pollution_kind, message)
                if not(target_site in result):
                    result[target_site] = dict()
                result[target_site][interval_hours] = ensemble_model.rnn(target_kind, local, city, target_site, training_year, testing_year,
                               training_duration, testing_duration, interval_hours, data, is_training)[0][0]
                print('site: %s %d hour result = %f' % (target_site.decode('utf8'), interval_hours, result[target_site][interval_hours]))

            # 北部
            paralPredict('北部', '台北', '古亭', target_kind, message, training_year, testing_year, training_duration,
                         testing_duration, 1, is_training, result)
            paralPredict('北部', '台北', '古亭', target_kind, message, training_year, testing_year, training_duration,
                         testing_duration, 6, is_training, result)
            paralPredict('北部', '台北', '古亭', target_kind, message, training_year, testing_year, training_duration,
                         testing_duration, 12, is_training, result)

            # 中部
            paralPredict('中部', '台中', '忠明', target_kind, message, training_year, testing_year, training_duration,
                         testing_duration, 1, is_training, result)
            paralPredict('中部', '台中', '忠明', target_kind, message, training_year, testing_year, training_duration,
                         testing_duration, 6, is_training, result)
            paralPredict('中部', '台中', '忠明', target_kind, message, training_year, testing_year, training_duration,
                         testing_duration, 12, is_training, result)

            # 高屏
            paralPredict('高屏', '高雄', '左營', target_kind, message, training_year, testing_year, training_duration,
                         testing_duration, 1, is_training, result)
            paralPredict('高屏', '高雄', '左營', target_kind, message, training_year, testing_year, training_duration,
                         testing_duration, 6, is_training, result)
            paralPredict('高屏', '高雄', '左營', target_kind, message, training_year, testing_year, training_duration,
                         testing_duration, 12, is_training, result)

            """
            # thread_pool = []
            # # 1 hour
            # thread_pool.append(threading.Thread(target=paralPredict, args=(
            #     '北部', '台北', '古亭', pollution_kind, message, training_year, testing_year, training_duration,
            #     testing_duration, 1, is_training, result)))
            # thread_pool.append(threading.Thread(target=paralPredict, args=(
            #     '中部', '台中', '忠明', pollution_kind, message, training_year, testing_year, training_duration,
            #     testing_duration, 1, is_training, result)))
            # thread_pool.append(threading.Thread(target=paralPredict, args=(
            #     '高屏', '高雄', '左營', pollution_kind, message, training_year, testing_year, training_duration,
            #     testing_duration, 1, is_training, result)))
            #
            # # 6 hour
            # thread_pool.append(threading.Thread(target=paralPredict, args=(
            #     '北部', '台北', '古亭', pollution_kind, message, training_year, testing_year, training_duration,
            #     testing_duration, 6, is_training, result)))
            # thread_pool.append(threading.Thread(target=paralPredict, args=(
            #     '中部', '台中', '忠明', pollution_kind, message, training_year, testing_year, training_duration,
            #     testing_duration, 6, is_training, result)))
            # thread_pool.append(threading.Thread(target=paralPredict, args=(
            #     '高屏', '高雄', '左營', pollution_kind, message, training_year, testing_year, training_duration,
            #     testing_duration, 6, is_training, result)))
            #
            # # 12 hour
            # thread_pool.append(threading.Thread(target=paralPredict, args=(
            #     '北部', '台北', '古亭', pollution_kind, message, training_year, testing_year, training_duration,
            #     testing_duration, 12, is_training, result)))
            # thread_pool.append(threading.Thread(target=paralPredict, args=(
            #     '中部', '台中', '忠明', pollution_kind, message, training_year, testing_year, training_duration,
            #     testing_duration, 12, is_training, result)))
            # thread_pool.append(threading.Thread(target=paralPredict, args=(
            #     '高屏', '高雄', '左營', pollution_kind, message, training_year, testing_year, training_duration,
            #     testing_duration, 12, is_training, result)))
            #
            # for i in range(len(thread_pool)):
            #     thread_pool[i].start()
            # for i in range(len(thread_pool)):
            #     thread_pool[i].join()
            """

            # example
            # result['左營'] = dict()
            # result['左營']['1'] = 0.1
            # result['左營']['6'] = 0.5
            # result['左營']['12'] = 0.75

            print(result)
            final_time = time.time()
            time_spent_printer(start_time, final_time)
            msg = json.dumps(result)

            # code of updating files in google drive
            # file_id = ''
            # # # Auto-iterate through all files that matches this query
            # file_list = drive.ListFile({'q': "title='Airdata.json' and trashed=false"}).GetList()
            # for file1 in file_list:
            #     print('title: %s, id: %s' % (file1['title'], file1['id']))
            #     file_id = file1['id']
            #
            # if file_id != '':
            #     file1 = drive.CreateFile({'id': file_id})
            #     file1.SetContentString(msg)  # Set content of the file from given string.
            #     file1.Upload()
            # else:
            #     file1 = drive.CreateFile({'title': 'Airdata.json',
            #                               'mimeType': 'application/json'})  # Create GoogleDriveFile instance with title 'Hello.txt'.
            #     file1.SetContentString(msg)  # Set content of the file from given string.
            #     file1.Upload()

            # -----------------------------------------------------------------------------
            print(msg)
            slave.send_json(msg)
            print('finish process: slave {0}'.format(name))

        except:
            print('slave {0} error'.format(name))
            msg = json.dumps(result)
            slave.send_json(msg)
            raise
            break
    slave.close()

slave_num = 3

context = zmq.Context()

router = context.socket(zmq.XREP)  # or zmq.ROUTER, their value are both 6
router.bind(url_router)

slaves = context.socket(zmq.XREQ)  # or zmq.DEALER, their value are both 5
slaves.bind(url_slave)

for i in range(slave_num):
    thread = threading.Thread(target=server_slave, args=(i, url_slave, context,))
    thread.start()

zmq.device(zmq.QUEUE, router, slaves)

router.close()
slaves.close()
context.term()
