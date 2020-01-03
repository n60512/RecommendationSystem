from DBconnector import DBConnection
import time

"""
This program is used to counting the number of REAL TIME PAIR (Setting reviewCount)
"""


#%% Load dataset from database
def loadData(conn):

    st = time.time()
    print('Loading dataset from database...') 

    sql = (
		'SELECT reviewerID '+
		'FROM clothing_review '+
		'group by reviewerID '+
		'HAVING COUNT(reviewerID) >= 15;'
    )

    
    res = conn.selection(sql)
    

    print('Loading complete. [{}]'.format(time.time()-st))

    st = time.time()
    userid = list()
    for row in res:
        userid.append(row['reviewerID'])

    print('Voc creation complete. [{}]'.format(time.time()-st))
    
    return userid


def SearchUser(conn, uid):
    sql = (
		'SELECT `ID`,reviewerID,unixReviewTime,reviewTime '+
		'FROM clothing_review '+
		'WHERE reviewerID=\'{}\' ORDER BY unixReviewTime, `ID`;'.format(uid)
    )    
    res = conn.selection(sql)

    return res

def realtimeExtration(res, reviewCount = 4, dayRange=3, unixPerDay=86400):

    _id = list()
    uid = list()
    unixtime = list()
    datetime = list()

    uid2realtAvg =  list()

    realtimels = list()

    for row in res:
        _id.append(row['ID'])
        uid.append(row['reviewerID'])
        unixtime.append(row['unixReviewTime'])
        datetime.append(row['reviewTime'])

    for i in range(len(uid)):
        if(i+reviewCount<=len(uid)):
            tmp = sum(unixtime[i:i+reviewCount]) / reviewCount
            tmpAvg = tmp-unixtime[i]
            uid2realtAvg.append(tmpAvg)

            if(tmpAvg <= dayRange*unixPerDay):
                realtimels.append(_id[i])

        else:
            uid2realtAvg.append(-1)
            
    debugMode = False
    if(debugMode):
        for i in range(len(uid)):
            print(_id[i], end=' ')
            print(uid[i], end=' ')
            print(unixtime[i], end=' ')
            print(datetime[i], end='\t')
            print(uid2realtAvg[i], end='\t')

            if(uid2realtAvg[i] <= dayRange*unixPerDay):
                print('v', end=' ')
            
            print()
        print(realtimels)

    return realtimels


if __name__ == "__main__":
        
    conn = DBConnection()
    userid = loadData(conn)

    ctr = 0

    for uid in userid:
        res = SearchUser(conn, uid)
        realtimels = realtimeExtration(res, reviewCount = 8)

        ctr += len(realtimels)

    print(ctr)

    conn.close()