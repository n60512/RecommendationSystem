#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

def clothing_compare():
    dir = dict()
    modelName = ['clothing_r4_pre_cat_lk32', 'clothing_r4_pre_cat_lk32_RSNRtraintest',
     'clothing_r4_pre_cat_lk32_RSNRtest']
        # 'NCF', 
        # 'clothing_r_pregru_cat', 'clothing_r_prencf_cat', 'clothing_r_nopre_cat', 'clothing_r_pre_cat_nolimit',
        # 'clothing_r_pre_cat_nolimit_lk32_t200']
        
    dir[modelName[0]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_cat_r4_bs40_lr1e05_lk32_dec20_dp0_nolimit_testset200_01/TestingLoss.txt'
    dir[modelName[1]] = R'ReviewsPrediction_Model/Loss/1209_clothing_pre_cat_r4_bs40_lr1e05_lk32_dec20_dp0_nolimit_testset200_01_traintestRSNR/TestingLoss.txt'
    dir[modelName[2]] = R'ReviewsPrediction_Model/Loss/1209_clothing_pre_cat_r4_bs40_lr1e05_lk32_dec20_dp0_nolimit_testset200_01_testRSNR/TestingLoss.txt'
    # dir[modelName[3]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_cat_r1_bs16_lr1e05_lk32_dec20_dp0_nolimit_testset200/TestingLoss.txt'
    

    
    # dir[modelName[3]] = R'ReviewsPrediction_Model/NCF/Loss/1201_clothing_ncf_bs16_lr1e05/TestingLoss.txt'
    # dir[modelName[4]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_cat_bs16_lr1e05/TestingLoss.txt'
    # dir[modelName[5]] = R'ReviewsPrediction_Model/Loss/1201_clothing_prencf_cat_bs16_lr1e05/TestingLoss.txt'
    # dir[modelName[6]] = R'ReviewsPrediction_Model/Loss/1201_clothing_nopre_cat_bs16_lr1e05/TestingLoss.txt'
    # dir[modelName[7]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_cat_bs16_lr1e05_dec20_dp5e01_nolimit/TestingLoss.txt'
    # dir[modelName[8]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_cat_bs16_lr1e05_lk32_dec20_dp5e01_nolimit_testset200/TestingLoss.txt'
    return dir, modelName

def clothing_bs_compare():
    dir = dict()
    modelName = [ 'HANN', 'interGeneral', 'allGeneral']
        
    dir[modelName[0]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_cat_r4_bs40_lr1e05_lk32_dec20_dp0_nolimit_testset200_01/TestingLoss.txt'
    dir[modelName[1]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_cat_r4_bs40_lr5e05_lk32_dec20_dp0_nolimit_testset200_interGeneral/TestingLoss.txt'
    dir[modelName[2]] = R'ReviewsPrediction_Model/Loss/1211_clothing_pre_cat_r4_bs40_lr1e05_lk32_dec20_dp0_nolimit_allGeneral/TestingLoss.txt'
    

    return dir, modelName

def clothing_reviews_compare():
    dir = dict()
    modelName = ['{}_reviews'.format(str(idx)) for idx in range(1, 10, 1)]

    for idx in range(1, 10, 1):
        
        dir[modelName[idx-1]] = R'ReviewsPrediction_Model/Loss/1201_clothing_pre_cat_r{}_bs16_lr1e05_lk32_dec20_dp5e01_nolimit_testset200/TestingLoss.txt'.format(idx)

    return dir, modelName

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

        if(len(modelList[mName])==25):
            model_loss[mName] = [index*2 for index in range(len(modelList[mName]))]
        else:
            model_loss[mName] = [index for index in range(len(modelList[mName]))]
        
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
    plt.plot([index+1 for index in range(len(minlist))], minlist ,'-o', label='loss')
    plt.legend(loc='best', framealpha=0.5, prop={'size': 'large', 'family': 'monospace'})


saveName = '1209_clothing_attn_compare'
fig = plt.figure(figsize=(12, 8), dpi=288)

# dir, modelName = clothing_compare()
dir, modelName = clothing_bs_compare()
# drawBest(dir)
drawLoss(dir)

fig.savefig(R'EvaluationImage/{}.png'.format(saveName), facecolor='w')
plt.clf()
