#!/usr/bin/env python
"""
GenerateColumn.py: Generate Electronic product and reviewrs Column.
"""
__author__      = "You-Xiang Chen"
__email__ = "n60512@gmail.com"

#%%
import math
import numpy as np
import time
from DBconnector import DBConnection

#%%
def SearchProductMetadata(conn):
    sql = (
        'SELECT metadata.`asin` ' +
        'FROM metadata ' +
        'WHERE metadata.`asin` in ' +
        '( ' +
        'SELECT DISTINCT(`asin`) ' +    # Only in Category
        'FROM review  ' +
        ');'
    )
    res = conn.selection(sql)
    return res

def SearchReviewerID(conn):
    sql = ('SELECT DISTINCT(`reviewerID`) ' +
        'FROM review ' +
        'ORDER BY reviewerID;'
        )
    res = conn.selection(sql)
    return res

def WriteRelate(idlist, fileName = 'reviewerID'):
    with open('{}.csv'.format(fileName), 'a') as file:
        for value in idlist:
            file.write(value + ',')

#%%
conn = DBConnection()
conn.connection

#%% Write reviewer id Column
res = SearchReviewerID(conn)
reviewerID = [res[i]['reviewerID'] for i in range(len(res))]
reviewerID[:10] 
WriteRelate(reviewerID)
#%% Write product asin Column
res = SearchProductMetadata(conn)
asin = [res[i]['asin'] for i in range(len(res))]
WriteRelate(asin, 'asin')

#%%
conn.close()

#%% For debug
with open(R'asin.csv', 'r') as file:
    content = file.read()
    content = content.split(',')
    print(len(content))
