# _*_ coding: utf-8 _*_

from __future__ import print_function

import zmq
import threading
import numpy as np
import time
import json

import rnn, reader

# target_program = 'test2.py'
url_slave = 'inproc://ping-workers'
# url_router = 'tcp://127.0.0.1:9527'
url_router = 'tcp://10.10.53.205:9527'


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
        try:
            message = slave.recv_json()
            print('slave {0} recv "{1}"'.format(name, message))
            start_time = time.time()

            training_year = '2014-2015'  # 2014-2015
            testing_year = '2015-2015'
            training_duration = '1/1-11/30'  # '1/1-11/30'
            testing_duration = '12/1-12/31'
            pollution_kind = ['PM2.5', 'O3', 'AMB_TEMP', 'RH', 'WIND_SPEED', 'WIND_DIREC']
            interval_hours = 6  # predict the label of average data of many hours later, default is 1
            is_training = False

            # ------------------------------------------------------------------------------
            result = dict()

            def paralPredict(local, city, target_site, pollution_kind, message, training_year, testing_year,
                               training_duration, testing_duration, interval_hours, is_training, result):
                data = reader.data_collection(local, city, pollution_kind, message)
                result[target_site] = rnn.rnn(pollution_kind, local, city, target_site, training_year, testing_year,
                               training_duration, testing_duration, interval_hours, data, is_training)[0][0]
                print('site: %s result = %f' % (target_site.decode('utf8'), result[target_site]))

            thread_pool = []
            thread_pool.append(threading.Thread(target=paralPredict, args=(
                '北部', '台北', '古亭', pollution_kind, message, training_year, testing_year, training_duration,
                testing_duration, interval_hours, is_training, result)))
            thread_pool.append(threading.Thread(target=paralPredict, args=(
                '中部', '台中', '豐原', pollution_kind, message, training_year, testing_year, training_duration,
                testing_duration, interval_hours, is_training, result)))
            thread_pool.append(threading.Thread(target=paralPredict, args=(
                '高屏', '高雄', '左營', pollution_kind, message, training_year, testing_year, training_duration,
                testing_duration, interval_hours, is_training, result)))
            for i in range(len(thread_pool)):
                thread_pool[i].start()
            for i in range(len(thread_pool)):
                thread_pool[i].join()

            # local = '北部'
            # city = '台北'
            # target_site = '古亭'
            # paralPredict(local, city, target_site, pollution_kind, message, training_year, testing_year,
            #              training_duration, testing_duration, interval_hours, is_training, result)

            # local = '中部'
            # city = '台中'
            # target_site = '豐原'
            # result[target_site] = paralPredict(local, city, target_site, pollution_kind, message,
            #                                                   training_year, testing_year, training_duration,
            #                                                   testing_duration, interval_hours, is_training)
            # print('site: %s result = %f' % (target_site, result[target_site]))
            #
            # local = '高屏'
            # city = '高雄'
            # target_site = '左營'
            # result[target_site] = paralPredict(local, city, target_site, pollution_kind, message,
            #                                                   training_year, testing_year, training_duration,
            #                                                   testing_duration, interval_hours, is_training)
            # print('site: %s result = %f' % (target_site, result[target_site]))

            print(result)
            final_time = time.time()
            time_spent_printer(start_time, final_time)
            msg = json.dumps(result)
            # -----------------------------------------------------------------------------
            print(msg)
            slave.send_json(msg)
            print('finish process: slave {0}'.format(name))

        except:
            print('slave {0} error'.format(name))
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
