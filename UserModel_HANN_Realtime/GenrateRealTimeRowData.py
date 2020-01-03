from DBconnector import DBConnection
import time

"""
This program is used to counting the number of REAL TIME PAIR (Setting reviewCount)
"""


#%% Load dataset from database
def loadUserid(conn, havingCount=15):

    st = time.time()
    print('Loading dataset from database...') 

    sql = (
		'SELECT reviewerID '+
		'FROM clothing_review '+
		'group by reviewerID '+
		'HAVING COUNT(reviewerID) >= {};'.format(havingCount)
    )
    res = conn.selection(sql)

    userid = list()
    for row in res:
        userid.append(row['reviewerID'])

    print('Loading user complete. ')

    return userid


def SearchUser(conn, uid):
    sql = (
		'SELECT `ID`,reviewerID,unixReviewTime,reviewTime '+
		'FROM clothing_review '+
		'WHERE reviewerID=\'{}\' ORDER BY unixReviewTime, `ID`;'.format(uid)
    )    
    res = conn.selection(sql)

    return res

def writeSqlFile(text):
    with open('realtime_8_8.sql','a') as f:
        f.writelines(text)

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
        # if list in the range of uid
        if(i+reviewCount<=len(uid)):
            tmp = sum(unixtime[i:i+reviewCount]) / reviewCount
            tmpAvg = tmp-unixtime[i]
            uid2realtAvg.append(tmpAvg)

            # Under day range
            if(tmpAvg <= dayRange*unixPerDay):
                realtimels.append(_id[i])
                
                for rank in range(reviewCount):
                    tmpSQL = (
                        'INSERT INTO clothing_realtime8_interaction8 (rank, `ID`, reviewerID) VALUES ({}, {}, \'{}\');\n'.format(
                            rank + 1, _id[i + rank], uid[i + rank]
                        )
                    )
                    writeSqlFile(tmpSQL)
                    stop =1


        else:
            # Out of range
            uid2realtAvg.append(-1)
            

    return realtimels


if __name__ == "__main__":
        
    conn = DBConnection()
    userid = loadUserid(conn, havingCount=8)

    ctr = 0

    for uid in userid:
        res = SearchUser(conn, uid)
        realtimels = realtimeExtration(res, reviewCount = 8)

        ctr += len(realtimels)

    print(ctr)

    conn.close()

# %%
