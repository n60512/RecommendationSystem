#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
import io

fig = plt.figure(figsize=(6, 4))
dir = dict()
modelName = ['clothing_r_nopre', 'clothing_r_pre', 'clothing_itemVec', 'NCF']
dir[modelName[0]] = R'ReviewsPrediction_Model/Loss/1127_clothing_nopre_bs4_lr5e07/TestingLoss.txt'
dir[modelName[1]] = R'ReviewsPrediction_Model/Loss/1127_clothing_pre_bs4_lr5e07/TestingLoss.txt'
dir[modelName[2]] = R'ReviewsPrediction_Model/UserItemFeature/Loss/1126_clothing_lr5e07/TestingLoss.txt'
dir[modelName[3]] = R'ReviewsPrediction_Model/NCF/Loss/1127_clothing_ncf_lr25e07/TestingLoss.txt'

#%%

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
# #%%
# multiple_errlist = list()
# with open(dir_2nd, 'r') as tf:
#     content = tf.readlines()
#     for line in content:
#         if('SE:' in line):    
#             err = float(line.split('RMSE:')[1].replace('\n',''))
#             multiple_errlist.append(err)

# #%%
# third_errlist = list()
# with open(dir_3rd, 'r') as tf:
#     content = tf.readlines()
#     for line in content:
#         if('SE:' in line):    
#             err = float(line.split('RMSE:')[1].replace('\n',''))
#             third_errlist.append(err)


#%%
# y_single = [index*2 for index in range(len(single_errlist))]
# y_multiple = [index*2 for index in range(len(multiple_errlist))]
# y_third = [index*2 for index in range(len(third_errlist))]


#%%

# plt.plot(y_single, single_errlist , label='clothing_nopre loss')
# plt.plot(y_multiple, multiple_errlist , label='clothing_pre loss')
# plt.plot(y_third, third_errlist , label='UI_Letent loss')



plt.legend(loc='best', framealpha=0.5, prop={'size': 'large', 'family': 'monospace'})

# plt.show()

fig.savefig(R'EvaluationImage/1127_clothing_compare_.png', facecolor='w')
plt.clf()