#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
import io

saveName = '1201_clothing_reviews_compare_test2'
fig = plt.figure(figsize=(12, 8), dpi=288)


def clothing_compare():
    dir = dict()
    modelName = ['clothing_r_nopre_nocat', 'clothing_r_pregru_nocat', 'clothing_itemVec', 'NCF', 
        'clothing_r_pregru_cat', 'clothing_r_prencf_cat', 'clothing_r_nopre_cat', 'clothing_r_pre_cat_nolimit',
        'clothing_r_pre_cat_nolimit_lk32_t200']
        
    dir[modelName[0]] = R'ReviewsPrediction_Model/Loss/1130_clothing_nopre_bs16_lr1e05/TestingLoss.txt'
    dir[modelName[1]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_bs16_lr1e05/TestingLoss.txt'
    dir[modelName[2]] = R'ReviewsPrediction_Model/UserItemFeature/Loss/1201_clothing_bs16_lr1e05/TestingLoss.txt'
    dir[modelName[3]] = R'ReviewsPrediction_Model/NCF/Loss/1201_clothing_ncf_bs16_lr1e05/TestingLoss.txt'
    dir[modelName[4]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_cat_bs16_lr1e05/TestingLoss.txt'
    dir[modelName[5]] = R'ReviewsPrediction_Model/Loss/1201_clothing_prencf_cat_bs16_lr1e05/TestingLoss.txt'
    dir[modelName[6]] = R'ReviewsPrediction_Model/Loss/1201_clothing_nopre_cat_bs16_lr1e05/TestingLoss.txt'

    dir[modelName[7]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_cat_bs16_lr1e05_dec20_dp5e01_nolimit/TestingLoss.txt'

    dir[modelName[8]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_cat_bs16_lr1e05_lk32_dec20_dp5e01_nolimit_testset200/TestingLoss.txt'
    return dir, modelName



def clothing_reviews_compare():
    dir = dict()
    modelName = ['reviews{}'.format(str(idx)) for idx in range(1, 10, 1)]

    for idx in range(1, 10, 1):
        
        dir[modelName[idx-1]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_cat_r{}_bs16_lr1e05_lk32_dec20_dp5e01_nolimit_testset200/TestingLoss.txt'.format(idx)

    return dir, modelName

dir, modelName = clothing_reviews_compare()
#%%

def drawLoss(dir):
    modelList = dict()
    model_loss = dict()
    for mName, fpath in dir.items():
        _errlist = list()
        with open(fpath, 'r') as tf:
            content = tf.readlines()
            for line in content:
                if('RMSE:' in line):    
                    err = float(line.split('RMSE:')[1].replace('\n',''))
                    _errlist.append(err)
        
        modelList[mName] = _errlist

        model_loss[mName] = [index*2 for index in range(len(modelList[mName]))]
        
        plt.plot(model_loss[mName], modelList[mName] , label='{} loss'.format(mName))


    plt.legend(loc='best', framealpha=0.5, prop={'size': 'large', 'family': 'monospace'})


def drawBest(dir):
    modelList = dict()
    model_loss = dict()

    minlist =list()

    for mName, fpath in dir.items():
        _errlist = list()
        with open(fpath, 'r') as tf:
            content = tf.readlines()
            for line in content:
                if('RMSE:' in line):    
                    err = float(line.split('RMSE:')[1].replace('\n',''))
                    _errlist.append(err)
        
        modelList[mName] = _errlist

        minlist.append(min(modelList[mName]))
    
    
    print(minlist)
    plt.plot([index for index in range(len(minlist))], minlist ,'-o', label='loss')

    #     model_loss[mName] = [index*2 for index in range(len(modelList[mName]))]
        
    #     plt.plot(model_loss[mName], modelList[mName] , label='{} loss'.format(mName))


    plt.legend(loc='best', framealpha=0.5, prop={'size': 'large', 'family': 'monospace'})


drawBest(dir)
# drawLoss(dir)

fig.savefig(R'EvaluationImage/{}.png'.format(saveName), facecolor='w')
plt.clf()

# %%
