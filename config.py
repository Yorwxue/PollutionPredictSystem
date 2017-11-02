# _*_ coding: utf-8 _*_
import os

config = dict()
config['root_path'] = os.path.dirname(os.path.abspath(__file__))  # '/home/csist/clliao/predictServer/'

config['python_ENV'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pm25control/bin/python3')
config['period'] = 12

"""
pollution_site_map = {
    '中部': {'台中': ['大里', '忠明', '沙鹿', '西屯', '豐原'],  # 5
           '南投': ['南投', '竹山'],  # 2
           '彰化': ['二林', '彰化']},  # 2

    '北部': {'台北': ['中山', '古亭', '士林', '松山', '萬華'],  # 5
           '新北': ['土城', '新店', '新莊', '板橋', '林口', '汐止', '菜寮', '萬里'],  # 8
           '基隆': ['基隆'],  # 1
           '桃園': ['大園', '平鎮', '桃園', '龍潭']}, # 4

    '宜蘭': {'宜蘭': ['冬山', '宜蘭']},  # 2

    '竹苗': {'新竹': ['新竹', '湖口', '竹東'],  # 3
           '苗栗': ['三義', '苗栗']},  # 2

    '花東': {'花蓮': ['花蓮'],  # 1
           '台東': ['臺東']},  # 1

    '北部離島': {'彭佳嶼': []},

    '西部離島': {'金門': ['金門'], # 1
             '連江': ['馬祖'],  # 1
             '東吉嶼': [],
             '澎湖': ['馬公']},  # 1

    '雲嘉南': {'雲林': ['崙背', '斗六', '竹山'],  # 3
            '台南': ['善化', '安南', '新營', '臺南'],  # 4
            '嘉義': ['嘉義', '新港', '朴子']},  # 3

    '高屏': {'高雄': ['仁武', '前金', '大寮', '小港', '左營', '林園', '楠梓', '美濃'],  # 8
           '屏東': ['屏東', '恆春', '潮州']}  # 3
}
"""

pollution_site_map = {
    '中部': {'台中': ['大里', '忠明', '沙鹿', '西屯', '豐原'],  # 5
           '南投': ['南投', '竹山', '埔里'],  # 2
           '彰化': ['彰化', '二林']},  # 2

    '北部': {'台北': ['中山', '古亭', '士林', '松山', '萬華'],  # 5
           '新北': ['林口', '土城', '新店', '新莊', '板橋', '汐止', '菜寮', '萬里'],  # 8
           '基隆': ['基隆'],  # 1
           '桃園': ['桃園', '大園', '平鎮', '龍潭']}, # 4

    '宜蘭': {'宜蘭': ['宜蘭', '冬山']},  # 2

    '竹苗': {'新竹': ['新竹', '湖口', '竹東'],  # 3
           '苗栗': ['苗栗', '三義']},  # 2

    '花東': {'花蓮': ['花蓮'],  # 1
           '台東': ['臺東']},  # 1

    '北部離島': {'彭佳嶼': []},

    '西部離島': {'金門': ['金門'], # 1
             '連江': ['馬祖'],  # 1
             '東吉嶼': [],
             '澎湖': ['馬公']},  # 1

    '雲嘉南': {'雲林': ['崙背', '斗六', '竹山'],  # 3
            '台南': ['臺南', '善化', '安南', '新營'],  # 4
            '嘉義': ['嘉義', '新港', '朴子']},  # 3

    '高屏': {'高雄': ['左營', '仁武', '前金', '大寮', '小港', '林園', '楠梓', '美濃'],  # 8
           '屏東': ['屏東', '恆春', '潮州']}  # 3
}


def get_config():
    return config


def site_map():
    return pollution_site_map
