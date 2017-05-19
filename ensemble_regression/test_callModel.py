# _*_ coding: utf-8 _*_

import ensemble_model
import numpy as np

training_year = '2014-2016'  # 2014-2015
testing_year = '2017-2017'
training_duration = '1/1-12/31'  # '1/1-11/30'
testing_duration = '1/1-1/31'
target_kind = 'PM2.5'
pollution_kind = ['PM2.5', 'O3', 'AMB_TEMP', 'RH', 'WIND_SPEED', 'WIND_DIREC']
interval_hours = 6  # predict the label of average data of many hours later, default is 1
is_training = False

local = '北部'
city = '桃園'
target_site = '龍潭'
data = np.random.uniform(0, 1, [12, 4 * len(pollution_kind)])

pred = ensemble_model.ensemble_model(target_kind, local, city, target_site, training_year, testing_year,
                                     training_duration, testing_duration, interval_hours, data, is_training)

print()
