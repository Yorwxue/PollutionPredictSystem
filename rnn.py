# _*_ coding: utf-8 _*_

# GPU command:
#     THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python script.py

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import time
import sys
import theano
import cPickle
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import DropoutLSTM
from keras.callbacks import ModelCheckpoint, ModelTest
from keras.regularizers import l2
import matplotlib.pyplot as plt

from reader import read_data_sets, construct_time_steps
from missing_value_processer import missing_check
from feature_processor import data_coordinate_angle

root_path = '/home/clliao/workspace/python/socket/rnn_regression/'

sys.path.insert(0, "/usr/local/cuda-7.5/bin")
sys.path.insert(0, root_path + "keras-BayesianRNN")  # point this to your local fork of https://github.com/yaringal/keras
sys.path.insert(0, "../Theano")

# Create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
print('Theano version: ' + theano.__version__ + ', base compile dir: '
      + theano.config.base_compiledir)
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False


def time_spent_printer(start_time, final_time):
    spent_time = final_time - start_time
    print('totally spent ', end='')
    print(int(spent_time / 3600), 'hours ', end='')
    print(int((int(spent_time) % 3600) / 60), 'minutes ', end='')
    print((int(spent_time) % 3600) % 60, 'seconds')


def target_level(target, kind='PM2.5'):
    # target should be a 1d-list
    if kind == 'PM2.5':
        if (target >= 0) and (target < 11.5):                # 0-11
            return 1
        elif (target >= 11.5) and (target < 23.5):           # 12-23
            return 2
        elif (target >= 23.5) and (target < 35.5):           # 24-35
            return 3
        elif (target >= 35.5) and (target < 41.5):           # 36-41
            return 4
        elif (target >= 41.5) and (target < 47.5):           # 42-47
            return 5
        elif (target >= 47.5) and (target < 53.5):           # 48-53
            return 6
        elif (target >= 53.5) and (target < 58.5):           # 54-58
            return 7
        elif (target >= 58.5) and (target < 64.5):           # 59-64
            return 8
        elif (target >= 64.5) and (target < 70.5):           # 65-70
            return 9
        elif target >= 70.5:                                                # others(71+)
            return 10
        else:
            input('error value: %d' % target)


pollution_site_map = {
    '中部': {'台中': ['大里', '忠明', '沙鹿', '西屯', '豐原'],
           '南投': ['南投', '竹山'],
           '彰化': ['二林', '彰化']},

    '北部': {'台北': ['中山', '古亭', '士林', '松山', '萬華'],
           '新北': ['土城', '新店', '新莊', '板橋', '林口', '汐止', '菜寮', '萬里'],
           '基隆': ['基隆'],
           '桃園': ['大園', '平鎮', '桃園', '龍潭']},

    '宜蘭': {'宜蘭': ['冬山', '宜蘭']},

    '竹苗': {'新竹': ['新竹', '湖口', '竹東'],
           '苗栗': ['三義', '苗栗']},

    '花東': {'花蓮': ['花蓮'],
           '台東': ['臺東']},

    '北部離島': {'彭佳嶼': []},

    '西部離島': {'金門': ['金門'],
             '連江': ['馬祖'],
             '東吉嶼': [],
             '澎湖': ['馬公']},

    '雲嘉南': {'雲林': ['崙背', '斗六'],
            '台南': ['善化', '安南', '新營', '臺南'],
            '嘉義': ['嘉義', '新港', '朴子']},

    '高屏': {'高雄': ['仁武', '前金', '大寮', '小港', '左營', '林園', '楠梓', '美濃'],
           '屏東': ['屏東', '恆春', '潮州']}
}


def rnn(pollution_kind, local, city, target_site, training_year, testing_year, training_duration, testing_duration, interval_hours, data, is_training):
    print('is_training(%s) = %s' % (target_site, is_training))
    # format of training_year and testing_year should be (start year)-(end year), like 2014-2015
    # format of training_duration and testing_duration should be (start date)-(end date), like 1/1-12/31

    # local = os.sys.argv[1]
    # city = os.sys.argv[2]
    site_list = pollution_site_map[local][city]

    # change format from   2014-2015   to   ['2014', '2015']
    training_year = [training_year[:training_year.index('-')], training_year[training_year.index('-')+1:]]
    testing_year = [testing_year[:testing_year.index('-')], testing_year[testing_year.index('-')+1:]]

    training_duration = [training_duration[:training_duration.index('-')], training_duration[training_duration.index('-')+1:]]
    testing_duration = [testing_duration[:testing_duration.index('-')], testing_duration[testing_duration.index('-')+1:]]
    interval_hours = int(interval_hours)  # predict the label of average data of many hours later, default is 1
    # is_training = os.sys.argv[9]   # True False

    # clear redundancy work
    if training_year[0] == training_year[1]:
        training_year.pop(1)
    if testing_year[0] == testing_year[1]:
        testing_year.pop(1)

    # Training Parameters
    # WIND_DIREC is a specific feature, that need to be processed, and it can only be element of input vector now.
    # pollution_kind = ['PM2.5', 'O3', 'AMB_TEMP', 'RH', 'WIND_SPEED', 'WIND_DIREC']
    target_kind = 'PM2.5'
    data_update = False
    # batch_size = 24 * 7
    seed = 0

    # Network Parameters
    input_size = (len(site_list)*len(pollution_kind)+len(site_list)) if 'WIND_DIREC' in pollution_kind else (len(site_list)*len(pollution_kind))
    time_steps = 12
    hidden_size = 20
    output_size = 1

    # print("Expected args: p_W, p_U, p_dense, p_emb, weight_decay, batch_size, maxlen")
    # print("Using default args:")
    param = ["", "0.5", "0.5", "0.5", "0.5", "1e-6", "128", "200"]
    # args = [float(a) for a in sys.argv[1:]]
    args = [float(a) for a in param[1:]]
    # print(args)
    p_W, p_U, p_dense, p_emb, weight_decay, batch_size, maxlen = args
    batch_size = int(batch_size)
    maxlen = int(maxlen)
    testing_month = testing_duration[0][:testing_duration[0].index('/')]
    folder = root_path+"model/%s/%s/" % (local, city)
    filename = ("sa_DropoutLSTM_pW_%.2f_pU_%.2f_pDense_%.2f_pEmb_%.2f_reg_%f_batch_size_%d_cutoff_%d_epochs_%s_%sm_%sh"
                % (p_W, p_U, p_dense, p_emb, weight_decay, batch_size, maxlen, target_site, testing_month, interval_hours))
    print(filename)

    if is_training:
        # reading data
        print('Reading data for %s .. ' % target_site)
        start_time = time.time()
        print('preparing training set for %s ..' % target_site)
        X_train = read_data_sets(sites=site_list+[target_site], date_range=np.atleast_1d(training_year),
                                 beginning=training_duration[0], finish=training_duration[-1],
                                 feature_selection=pollution_kind, update=data_update)
        X_train = missing_check(X_train)
        Y_train = np.array(X_train)[:, -len(pollution_kind):]
        Y_train = Y_train[:, pollution_kind.index(target_kind)]
        X_train = np.array(X_train)[:, :-len(pollution_kind)]

        print('preparing testing set for %s..' % target_site)
        X_test = read_data_sets(sites=site_list + [target_site], date_range=np.atleast_1d(testing_year),
                                beginning=testing_duration[0], finish=testing_duration[-1],
                                feature_selection=pollution_kind, update=data_update)
        Y_test = np.array(X_test)[:, -len(pollution_kind):]
        Y_test = Y_test[:, pollution_kind.index(target_kind)]
        X_test = missing_check(np.array(X_test)[:, :-len(pollution_kind)])

        final_time = time.time()
        print('Reading data for %s.. ok, ' % target_site, end='')
        time_spent_printer(start_time, final_time)

        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')

        if (len(X_train) < time_steps) or (len(X_test) < time_steps):
            input('time_steps(%d) too long.' % time_steps)

        # normalize
        print('Normalize for %s ..' % target_site)
        mean_X_train = np.mean(X_train, axis=0)
        std_X_train = np.std(X_train, axis=0)
        if 0 in std_X_train:
            input("Denominator can't be 0.(%s)" % target_site)
        X_train = np.array([(x_train-mean_X_train)/std_X_train for x_train in X_train])
        X_test = np.array([(x_test-mean_X_train)/std_X_train for x_test in X_test])

        mean_y_train = np.mean(Y_train)
        std_y_train = np.std(Y_train)
        if not std_y_train:
            input("Denominator can't be 0.(%s)" % target_site)
        Y_train = [(y - mean_y_train) / std_y_train for y in Y_train]
        print('mean_y_train: %f  std_y_train: %f (%s)' % (mean_y_train, std_y_train, target_site))

        fw = open(folder + filename + ".pickle", 'wb')
        cPickle.dump(str(mean_X_train) + ',' +
                     str(std_X_train) + ',' +
                     str(mean_y_train) + ',' +
                     str(std_y_train), fw)
        fw.close()

        # feature process
        if 'WIND_DIREC' in pollution_kind:
            index_of_kind = pollution_kind.index('WIND_DIREC')
            length_of_kind_list = len(pollution_kind)
            len_of_sites_list = len(site_list)
            X_train = X_train.tolist()
            X_test = X_test.tolist()
            for i in range(len(X_train)):
                for j in range(len_of_sites_list):
                    specific_index = index_of_kind + j * length_of_kind_list
                    coordin = data_coordinate_angle((X_train[i].pop(specific_index+j))*std_X_train[specific_index]+mean_X_train[specific_index])
                    X_train[i].insert(specific_index, coordin[1])
                    X_train[i].insert(specific_index, coordin[0])
                    if i < len(X_test):
                        coordin = data_coordinate_angle((X_test[i].pop(specific_index+j))*std_X_train[specific_index]+mean_X_train[specific_index])
                        X_test[i].insert(specific_index, coordin[1])
                        X_test[i].insert(specific_index, coordin[0])
            X_train = np.array(X_train)
            X_test = np.array(X_test)
        Y_test = np.array(Y_test, dtype=np.float)

        # --
        print('Constructing time series data set for %s ..' % target_site)
        X_train = construct_time_steps(X_train[:-1], time_steps)
        Y_train = Y_train[time_steps:]
        reserve_hours = interval_hours - 1
        deadline = 0
        for i in range(len(Y_train)):
            # check the reserve data is enough or not
            if (len(Y_train)-i-1) < reserve_hours:
                deadline = i
                break  # not enough
            for j in range(reserve_hours):
                Y_train[i] += Y_train[i+j+1]
            Y_train[i] /= interval_hours
        if deadline:
            X_train = X_train[:deadline]
            Y_train = Y_train[:deadline]

        X_test = construct_time_steps(X_test[:-1], time_steps)
        Y_test = Y_test[time_steps:]
        deadline = 0
        for i in range(len(Y_test)):
            # check the reserve data is enough or not
            if (len(Y_test)-i-1) < reserve_hours:
                deadline = i
                break  # not enough
            for j in range(reserve_hours):
                Y_test[i] += Y_test[i+j+1]
            Y_test[i] /= interval_hours
        if deadline:
            X_test = X_test[:deadline]
            Y_test = Y_test[:deadline]

        # delete data which have missing values
        i = 0
        while i < len(Y_test):
            if not(Y_test[i] > -10000):  # check missing or not, if Y_test[i] is missing, then this command will return True
                Y_test = np.delete(Y_test, i, 0)
                X_test = np.delete(X_test, i, 0)
                i = -1
            i += 1
        Y_test = np.array(Y_test, dtype=np.float)
        # --
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)

        np.random.seed(seed)
        np.random.shuffle(X_train)
        np.random.seed(seed)
        np.random.shuffle(Y_train)

    # ------------------------------------
    else:
        fr = open(folder + filename + ".pickle", 'rb')
        [mean_X_train, std_X_train, mean_y_train, std_y_train] = (cPickle.load(fr)).split(',')
        mean_X_train = mean_X_train.replace('[', '').replace(']', '').replace('\n', '').split(' ')
        while '' in mean_X_train:
            mean_X_train.pop(mean_X_train.index(''))
        mean_X_train = np.array(mean_X_train, dtype=np.float)
        std_X_train = std_X_train.replace('[', '').replace(']', '').replace('\n', '').split(' ')
        while '' in std_X_train:
            std_X_train.pop(std_X_train.index(''))
        std_X_train = np.array(std_X_train, dtype=np.float)
        mean_y_train = float(mean_y_train)
        std_y_train = float(std_y_train)
        fr.close()

        # input data
        X_test = data

        # normalize
        print('Normalize for %s ..' % target_site)
        X_test = np.array([(x_test - mean_X_train) / std_X_train for x_test in X_test])

        # feature process
        if 'WIND_DIREC' in pollution_kind:
            index_of_kind = pollution_kind.index('WIND_DIREC')
            length_of_kind_list = len(pollution_kind)
            len_of_sites_list = len(site_list)
            X_test = X_test.tolist()
            for i in range(len(X_test)):
                for j in range(len_of_sites_list):
                    specific_index = index_of_kind + j * length_of_kind_list
                    coordin = data_coordinate_angle(
                        (X_test[i].pop(specific_index + j)) * std_X_train[specific_index] + mean_X_train[
                            specific_index])
                    X_test[i].insert(specific_index, coordin[1])
                    X_test[i].insert(specific_index, coordin[0])
            X_test = np.array([X_test])

    print('Build model for %s ..' % target_site)
    start_time = time.time()
    model = Sequential()
    model.add(DropoutLSTM(input_size, hidden_size, truncate_gradient=maxlen, W_regularizer=l2(weight_decay),
                          U_regularizer=l2(weight_decay),
                          b_regularizer=l2(weight_decay),
                          p_W=p_W, p_U=p_U))
    model.add(Dropout(p_dense))
    model.add(Dense(hidden_size, output_size, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay)))

    # optimiser = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
    optimiser = 'adam'
    model.compile(loss='mean_squared_error', optimizer=optimiser)
    final_time = time.time()
    time_spent_printer(start_time, final_time)

    # --

    if is_training:
        print("Train for %s .." % target_site)
        start_time = time.time()
        checkpointer = ModelCheckpoint(filepath=folder+filename+".hdf5",
            verbose=1, append_epoch_name=False, save_every_X_epochs=50)
        modeltest_1 = ModelTest(X_train[:100], mean_y_train + std_y_train * np.atleast_2d(Y_train[:100]).T,
                                test_every_X_epochs=1, verbose=0, loss='euclidean',
                                mean_y_train=mean_y_train, std_y_train=std_y_train, tau=0.1)
        modeltest_2 = ModelTest(X_test, np.atleast_2d(Y_test).T, test_every_X_epochs=1, verbose=0, loss='euclidean',
                                mean_y_train=mean_y_train, std_y_train=std_y_train, tau=0.1)
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=251,
                  callbacks=[checkpointer, modeltest_1, modeltest_2])
        # score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
        # print('Test score:', score)
        # print('Test accuracy:', acc)

        # model.save_weights(folder+filename+"_250.hdf5", overwrite=True)
        final_time = time.time()
        time_spent_printer(start_time, final_time)

        # --

        print("Test for %s .." % target_site)
        standard_prob = model.predict(X_train, batch_size=500, verbose=1)
        print(np.mean(((mean_y_train + std_y_train * np.atleast_2d(Y_train).T)
                       - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5)

        # --

        standard_prob = model.predict(X_test, batch_size=500, verbose=1)
        T = 50
        prob = np.array([model.predict_stochastic(X_test, batch_size=500, verbose=0)
                         for _ in xrange(T)])
        prob_mean = np.mean(prob, 0)
        print(np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5)
        print(np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * prob_mean))**2, 0)**0.5)


        standard_prob_pred = np.zeros(len(standard_prob))
        prob_mean_pred = np.zeros(len(prob_mean))
        real_target = np.zeros(len(Y_test))

        standard_prob_true = 0.
        standard_prob_false = 0.
        prob_mean_true = 0.
        prob_mean_false = 0.

        # calculate the accuracy of ten level
        for i in range(len(prob_mean)):
            standard_prob_pred[i] = target_level(mean_y_train + std_y_train * prob_mean[i])
            prob_mean_pred[i] = target_level(mean_y_train + std_y_train * prob_mean[i])
            real_target[i] = target_level(Y_test[i])

            if real_target[i] == standard_prob_pred[i]:
                standard_prob_true += 1
            else:
                standard_prob_false += 1

            if real_target[i] == prob_mean_pred[i]:
                prob_mean_true += 1
            else:
                prob_mean_false += 1

        print('standard_prob_accuracy(%s): %.5f' % (target_site, standard_prob_true / ((standard_prob_true + standard_prob_false))))
        print('prob_mean_accuracy(%s): %.5f' % (target_site, (prob_mean_true / (prob_mean_true + prob_mean_false))))

        print('--')

        ha = 0.0  # observation high, predict high
        hb = 0.0  # observation low, predict high
        hc = 0.0  # observation high, predict low
        hd = 0.0  # observation low, predict low
        vha = 0.0  # observation very high, predict very high
        vhb = 0.0
        vhc = 0.0
        vhd = 0.0
        two_label_true = 0.0
        two_label_false = 0.0
        # statistic of status of prediction by forecast & observation
        for each_label in np.arange(len(real_target)):
            if real_target[each_label] >= 7:  # observation high
                if prob_mean_pred[each_label] >= 7:
                    ha += 1
                    two_label_true += 1
                else:
                    hc += 1
                    two_label_false += 1
            else:  # observation low
                if prob_mean_pred[each_label] >= 7:
                    hb += 1
                    two_label_false += 1
                else:
                    hd += 1
                    two_label_true += 1

            if real_target[each_label] >= 10:  # observation very high
                if prob_mean_pred[each_label] >= 10:
                    vha += 1
                else:
                    vhc += 1
            else:  # observation low
                if prob_mean_pred[each_label] >= 10:
                    vhb += 1
                else:
                    vhd += 1

        print('Two level accuracy of %s : %f' % (target_site, (two_label_true / (two_label_true + two_label_false))))
        print('high label of %s: (%d, %d, %d, %d)' % (target_site, ha, hb, hc, hd))
        print('very high label of %s: (%d, %d, %d, %d)' % (target_site, vha, vhb, vhc, vhd))

        # plot the real trend and trend of prediction
        prediction = mean_y_train + std_y_train * prob_mean
        plt.plot(np.arange(len(prediction)), Y_test[:len(prediction)], c='gray')
        plt.plot(np.arange(len(prediction)), prediction, color='pink')

        plt.xticks(np.arange(0, len(prediction), 24))
        plt.yticks(np.arange(0, max(Y_test), 10))
        plt.grid(True)
        plt.rc('axes', labelsize=4)

    else:
        print('loading model for %s ..' % target_site)
        model.load_weights(folder + filename + ".hdf5")

        standard_prob = model.predict(X_test, batch_size=1, verbose=1)
        T = 50
        prob = np.array([model.predict_stochastic(X_test, batch_size=1, verbose=0)
                         for _ in xrange(T)])
        prob_mean = np.mean(prob, 0)

    return mean_y_train + std_y_train * prob_mean
