#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
import io

singleGRU_directory = R'ReviewsPrediction_Model/SingleIntraGRU/Loss/1122_toys_lr5e07/TestingLoss.txt'
multipleGRU_directory = R'ReviewsPrediction_Model/Loss/1113_toys_lr1e06/TestingLoss.txt'

#%%
single_errlist = list()
with open(singleGRU_directory, 'r') as tf:
    content = tf.readlines()
    for line in content:
        if('RMSE:' in line):    
            err = float(line.split('RMSE:')[1].replace('\n',''))
            single_errlist.append(err)

#%%
multiple_errlist = list()
with open(multipleGRU_directory, 'r') as tf:
    content = tf.readlines()
    for line in content:
        if('SE:' in line):    
            err = float(line.split('RMSE:')[1].replace('\n',''))
            multiple_errlist.append(err)


#%%
y_single = [index*2 for index in range(len(single_errlist))]
y_multiple = [index*2 for index in range(len(multiple_errlist))]


#%%
fig = plt.figure(figsize=(6, 4))
plt.plot(y_single, single_errlist , label='y_single loss')
plt.plot(y_multiple, multiple_errlist , label='y_multiple loss')



plt.legend(loc='best', framealpha=0.5, prop={'size': 'large', 'family': 'monospace'})

# plt.show()

fig.savefig(R'toys_Svm_Loss.png', facecolor='w')
plt.clf()