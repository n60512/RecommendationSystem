"""
@ Generate product multi-hot encoding from Database . (multi-Process version)
@ Store encoded file at GenerateMultihotFeatureEmbedding\multihotEncode_item\fileName.txt
"""
#%%
from DBconnector import DBConnection
import pandas as pd
import time
import threading
import multiprocessing
import json

#%%
def SearchProductMetadata(conn):
    sql = (
        'SELECT metadata.`asin`, metadata.related, metadata.categories ' +
        'FROM metadata ' +
        'WHERE metadata.`asin` in ' +
        '( ' +
        'SELECT DISTINCT(`asin`) ' +    # Only in Category
        'FROM review  ' +
        ');'
    )
    res = conn.selection(sql)
    return res

#%%
def WriteRelate(related_dict, fileName = 'AlsoViewed'):
    with open('Related_{}.csv'.format(fileName), 'a') as file:
        for key, value in related_dict.items():
            file.write(key + ',')

#%% Generate product's related id list.
def GenerateRelatedIDList(productMetadata):

    count = 1
    alsoViewed = {}
    alsoBought = {}
    buyAfterViewing = {}
    boughtTogether = {}

    for product in productMetadata:
        # Show percent of process
        if(len(productMetadata)%count == 0):
            print('count : {}'.format(count))
        count += 1

        try:
            related = str(product['related']).replace('\'','\"')

            if(related != 'None'):
                # Check if related obj is exist
                jsonObj = json.loads(related)

                if('also_viewed' in jsonObj):   
                    alsoViewedObj = jsonObj['also_viewed']
                    for asin in alsoViewedObj:
                        alsoViewed[asin] = alsoViewed.get(asin, 0) + 1

                if('also_bought' in jsonObj):   
                    alsoBoughtObj = jsonObj['also_bought']
                    for asin in alsoBoughtObj:
                        alsoBought[asin] = alsoBought.get(asin, 0) + 1
                
                if('buy_after_viewing' in jsonObj):   
                    buyAfterViewingObj = jsonObj['buy_after_viewing']
                    for asin in buyAfterViewingObj:
                        buyAfterViewing[asin] = buyAfterViewing.get(asin, 0) + 1
                
                if('bought_together' in jsonObj):   
                    boughtTogetherObj = jsonObj['bought_together']
                    for asin in boughtTogetherObj:
                        boughtTogether[asin] = boughtTogether.get(asin, 0) + 1

        except Exception as msg:
            print('Exception ... asin : {}'.format(product['asin']))
            break
    
    return alsoViewed, alsoBought, buyAfterViewing, boughtTogether

#%%
"""
@related_behavior : alsoViewed, alsoBought, buyAfterViewing, boughtTogether list
@jsonObj : this item's jsonObj
"""
def GenerateRelateMultihotEncode(jsonObj ,related_behavior, behavior_text):
    multihotContent = ''

    if(behavior_text in jsonObj):
        AlsoViewedObj = jsonObj[behavior_text]

        for asin in related_behavior:
            flag = True
            for his_asin in AlsoViewedObj:
                if(his_asin == asin):
                    multihotContent = multihotContent + '1,'
                    flag = False
            if(flag):
                multihotContent = multihotContent + '0,'

    if(multihotContent ==  ''):
        tmpContent = ['0' for tmp in range(len(related_behavior))]
        return ','.join(tmpContent)+','
    else:
        return multihotContent

#%%
def CHECK_ENCODE_DEBUG(alsoViewed_Multihot, alsoBought_Multihot, buyAfterViewing_Multihot, boughtTogether_Multihot, itemRelated_Multihot):

    if(alsoViewed_Multihot != ''):
        counter = [int(val) for val in alsoViewed_Multihot.split(',')]
        counter = sum(counter)
        print('AV counter : {} len : {}'.format(counter, len(alsoViewed_Multihot.split(','))))

    if(alsoBought_Multihot != ''):
        counter = [int(val) for val in alsoBought_Multihot.split(',')]
        counter = sum(counter)
        print('AB counter : {} len : {}'.format(counter, len(alsoBought_Multihot.split(','))))

    if(buyAfterViewing_Multihot != ''):
        counter = [int(val) for val in buyAfterViewing_Multihot.split(',')]
        counter = sum(counter)
        print('BAV counter : {} len : {}'.format(counter, len(buyAfterViewing_Multihot.split(','))))

    if(boughtTogether_Multihot != ''):
        counter = [int(val) for val in boughtTogether_Multihot.split(',')]
        counter = sum(counter)
        print('BT counter : {} len : {}'.format(counter, len(boughtTogether_Multihot.split(','))))

    counter = [int(val) for val in itemRelated_Multihot.split(',')]
    counter = sum(counter)
    print('itemEcd. counter : {} len : {}'.format(counter, len(itemRelated_Multihot.split(','))))            

#%%
def GenerateJob(related_behavior, productMetadata):
    count = 1
    st = time.time()
    for product in productMetadata[:]:
        # Show percent of process
        if(len(productMetadata)%count == 0 and False):
            print('count : {}\tCost time : {}'.format(count, (time.time()-st) ))
            st = time.time()
        count += 1

        related = str(product['related']).replace('\'','\"')
        if(related != 'None'):
            # Check if related obj is exist
            jsonObj = json.loads(related)

            # generate relate encoding ; remove last ,
            alsoViewed, alsoBought, boughtTogether, buyAfterViewing = related_behavior

            alsoViewed_Multihot = GenerateRelateMultihotEncode(jsonObj, alsoViewed, 'also_viewed')[:-1]
            alsoBought_Multihot = GenerateRelateMultihotEncode(jsonObj, alsoBought, 'also_bought')[:-1]
            boughtTogether_Multihot = GenerateRelateMultihotEncode(jsonObj, boughtTogether, 'bought_together')[:-1]
            buyAfterViewing_Multihot = GenerateRelateMultihotEncode(jsonObj, buyAfterViewing, 'buy_after_viewing')[:-1]
            
            """
            alsoViewed count : 139008
            alsoBought count : 228226
            boughtTogether count : 17409
            buyAfterViewing count : 54404
            """
            itemRelated_Multihot = ','.join(
                [alsoViewed_Multihot, alsoBought_Multihot, 
                boughtTogether_Multihot, buyAfterViewing_Multihot]
            )

            # Check if multihot encoding has value
            CHECK_ENCODE = True
            if(not CHECK_ENCODE):
                print('======================')
                print('asin : {}'.format(product['asin']))
                CHECK_ENCODE_DEBUG(alsoViewed_Multihot, alsoBought_Multihot, buyAfterViewing_Multihot, boughtTogether_Multihot, itemRelated_Multihot)

            # Wheater write into txt file (massive storege cost!)
            writefile = True
            if(writefile):
                with open(R'GenerateMultihotFeatureEmbedding\multihotEncode_item\{}.txt'.format(product['asin']), 'a') as file:
                    file.write(itemRelated_Multihot)
    
    # print('Final count : {}'.format(count))

#%% Multi-Process version
def main(related_behavior, productMetadata, PROCESSOR = 6):

    print('GenerateJob Start')
    processes = []
    RANGE = len(productMetadata)
    step = int(RANGE/PROCESSOR)

    for index in range(step, RANGE + 1, step):    # RANGE +1 to avoid last step loop !
        p = multiprocessing.Process(
            target = GenerateJob,
            args=(related_behavior, 
            productMetadata[index-step:index - 1])
            )
        processes.append(p)
        p.start()

    # IF last loop out of reviewer's index
    if(RANGE%PROCESSOR > 0):
        p = multiprocessing.Process(target = GenerateJob,
            args=(related_behavior, 
            productMetadata[index:])
            )
        processes.append(p)
        p.start()
    
    # Wait for process finish
    for process in processes:
        process.join()
 
    print('Finish.')

if __name__ == "__main__":
    #%% Select for product metadata
    print('sql select start.')
    conn = DBConnection()
    productMetadata = SearchProductMetadata(conn)
    conn.close()
    print('sql select finish.')

    alsoViewed, alsoBought, buyAfterViewing, boughtTogether = GenerateRelatedIDList(productMetadata)
    
    alsoViewed = [*alsoViewed.keys()]
    alsoBought = [*alsoBought.keys()]
    buyAfterViewing = [*buyAfterViewing.keys()]
    boughtTogether = [*boughtTogether.keys()]

    related_behavior = (alsoViewed, alsoBought, buyAfterViewing, boughtTogether)

    # Start Process
    main(related_behavior, productMetadata[:])

    pass

#%%
"""
#%% Select for product metadata
print('sql select start.')
conn = DBConnection()
productMetadata = SearchProductMetadata(conn)
conn.close()
print('sql select finish.')

#%%
alsoViewed, alsoBought, buyAfterViewing, boughtTogether = GenerateRelatedIDList(productMetadata)
#%%
if(False):
    WriteRelate(alsoViewed, fileName = 'AlsoViewed')
    WriteRelate(alsoBought, fileName = 'AlsoBought')
    WriteRelate(buyAfterViewing, fileName = 'BuyAfterViewing')
    WriteRelate(boughtTogether, fileName = 'BoughtTogether')

#%%
alsoViewed = [*alsoViewed.keys()]
alsoBought = [*alsoBought.keys()]
buyAfterViewing = [*buyAfterViewing.keys()]
boughtTogether = [*boughtTogether.keys()]

"""