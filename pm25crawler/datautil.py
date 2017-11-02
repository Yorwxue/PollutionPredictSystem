import pymysql
import datetime

def pushNull(str1,str2,str3):
    if str3 == '' or str3 == 'ND':
        NowTime = datetime.datetime.now()
        minushour = datetime.timedelta(hours=12)
        WishTime = NowTime - minushour
        strWishTime = WishTime.strftime("%Y-%m-%d %H:00")
        NewNowTime = NowTime.strftime("%Y-%m-%d %H:00")
        conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='icrd00', charset='UTF8', db='AirData', cursorclass=pymysql.cursors.DictCursor)
        cursor = conn.cursor()  # 獲取一個游標對象
        sqlpull = "SELECT "+str2+" FROM AirDataTable Where PublishTime Between '"+strWishTime+"' AND '"+NewNowTime+"' AND SiteName = '"+str1+"'"
        try:
            cursor.execute(sqlpull)
            results = cursor.fetchall()
            counter = 0
            sum = 0
            for key in range(len(results)):
                if results[key][str2] != '' or results[key][str2] != 'NaN':
                    counter = counter + 1
                    sum = float(results[key][str2]) + sum
            if counter == 0:
                return str('NaN')
            sum = sum/counter
            strsum = str(sum)

            # json_results = str(resultDicts)


            conn.commit()  # 向資料庫中提交任何未解決的事務，對不支持事務的資料庫不進行任何操作
            conn.close()  # 關閉到資料庫的連接，釋放資料庫資源

            return (strsum)
        except Exception as g:
            print(g)

    else:
        return str(str3)

def CheckDatabaseUpdate():

    NowTime = datetime.datetime.now()
    minushour = datetime.timedelta(hours=1)
    WishTime = NowTime - minushour
    strWishTime = WishTime.strftime("%Y-%m-%d %H:00")
    NewNowTime = NowTime.strftime("%Y-%m-%d %H:00")
    conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='icrd00', charset='UTF8', db='AirData')
    cursor = conn.cursor()  # 獲取一個游標對象
    sqlpull = "SELECT PublishTime FROM AirDataTable Where PublishTime = '"+NewNowTime+"'"
    try:
        cursor.execute(sqlpull)
        results = cursor.fetchall()
        conn.commit()  # 向資料庫中提交任何未解決的事務，對不支持事務的資料庫不進行任何操作
        conn.close()  # 關閉到資料庫的連接，釋放資料庫資源
        if len(results) == 0:
            return False
        else:
            return True
        # return (strsum)
    except Exception as g:
        print(g)


