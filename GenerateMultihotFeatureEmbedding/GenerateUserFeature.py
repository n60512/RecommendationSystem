"""
@ Generate user multi-hot encoding from Database . (multi-Process version)
@ Store encoded file at GenerateMultihotFeatureEmbedding\multihotEncode_User\fileName.txt
"""
#%%
from DBconnector import DBConnection
import pandas as pd
import time
import threading
import multiprocessing

import torch
import torch.nn as nn

#%% Setting device
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

#%%
def SearchProduct(conn, table = 'cate'):

    if(table == 'metadata'):
        sql = ('SELECT `asin` FROM metadata;')
    elif(table == 'cate'):
        sql = ('SELECT DISTINCT(`asin`) FROM review;')
    res = conn.selection(sql)

    return res

def SearchReviewer(conn):

    sql = ('SELECT DISTINCT(reviewerID) ' +
            'FROM review ' +
            'ORDER BY reviewerID ;')
    res = conn.selection(sql)

    return res

def SearchReviewerData(conn):
    sql = ('SELECT `reviewerID`,`asin` ' +
        'FROM review ' +
        'ORDER BY reviewerID ;'
    )
    res = conn.selection(sql)

    return res

def WriteProductid():
    with open('productid.csv', 'a') as file:
        for row in productid:
            file.write(row['asin'] + ',')


#%%
def GenerateMultihotEncode(pd_reviewerdata, reviewerid, productid_list):

    df_Reviewer = pd_reviewerdata.loc[pd_reviewerdata['reviewerID'] == reviewerid]

    boughtHistory = list()
    multihotContent = ''

    for proid in df_Reviewer['asin']:
        boughtHistory.append(proid)

    for val in productid_list:
        flag = True
        for proid_ in boughtHistory:
            if(proid_ == val):
                multihotContent = multihotContent + '1,'
                flag = False
        if(flag):
            multihotContent = multihotContent + '0,'
    
    return multihotContent

#%% Genrating multihot encode
def GenerateJob(reviewerdata, reviewerid, productid_list, embedding = None):
    
    pd_reviewerdata = pd.DataFrame.from_dict(reviewerdata[:])

    for reviewerid_ in reviewerid[:]:
        st = time.time()    # time for testing

        # This reviewer's consuming history `multihot encoding`
        multihotEncode = GenerateMultihotEncode(pd_reviewerdata, 
            reviewerid_['reviewerID'], productid_list)[:-1]
        
        # Wheater write into txt file (massive storege cost!)
        writefile = True
        if(writefile):
            with open(R'GenerateMultihotFeatureEmbedding\multihotEncode_User\{}.txt'.format(reviewerid_['reviewerID']), 'a') as file:
                file.write(multihotEncode)
        
        # Construct embedding
        embed = False
        if(embed):
            multihotEncode = multihotEncode.split(',')
            multihotEncode = [int(i) for i in multihotEncode]
            torch_multihotEncode = torch.LongTensor([multihotEncode])
            torch_multihotEncode.to(device)
            embedded = embedding(torch_multihotEncode)
        
        print('Writing : {} time : {}'.format(reviewerid_['reviewerID'], (time.time()-st)))
    

#%%

def main(reviewerdata, reviewerid, productid_list, embedding = None):

    processes = []
    PROCESSOR = 8           # Working processor count
    RANGE = len(reviewerid) # Reviewrs count, len(reviewerid) = 192,403
    step = int(RANGE/PROCESSOR)+1

    print('Reviewr Length : {}'.format(RANGE))

    for index in range(step, RANGE, step):
        p = multiprocessing.Process(target = GenerateJob,
            args=(reviewerdata, reviewerid[index-step:index],
             productid_list, embedding))
        processes.append(p)
        p.start()

    # IF last loop out of reviewer's index
    if(RANGE%PROCESSOR > 0):
        p = multiprocessing.Process(target = GenerateJob,
            args=(reviewerdata, reviewerid[index:RANGE],
             productid_list, embedding))
        processes.append(p)
        p.start()        
        
    for process in processes:
        process.join()
 
    print('Finish.')

#%%
if __name__ == "__main__":
    
    print('sql select start.')
    conn = DBConnection()
    
    productid = SearchProduct(conn)
    reviewerid = SearchReviewer(conn)
    reviewerdata = SearchReviewerData(conn) 

    conn.close()
    print('sql select finish.')

    productid_list = [row['asin'] for row in productid]

    # # Initialize embeddings
    # hidden_size = 300
    # embedding = nn.Embedding(len(productid_list), hidden_size)
    
    main(reviewerdata, reviewerid, productid_list)

