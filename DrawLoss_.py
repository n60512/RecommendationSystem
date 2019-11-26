#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
import io

directory = R'ReviewsPrediction_Model/Loss/1114_elec_lr5e07'

#%%
testf_errlist = list()
with open(R'{}/TestingLoss.txt'.format(directory), 'r') as tf:
    content = tf.readlines()
    for line in content:
        if('RMSE:' in line):    
            err = float(line.split('RMSE:')[1].replace('\n',''))
            testf_errlist.append(err)

#%%
trainf_errlist = list()
with open(R'{}/TrainingLoss.txt'.format(directory), 'r') as tf:
    content = tf.readlines()
    for line in content:
        if('SE:' in line):    
            err = float(line.split('SE:')[1].replace('\n',''))
            trainf_errlist.append(err)
# %%
val_errlist = list()
with open(R'{}/ValidationLoss.txt'.format(directory), 'r') as tf:
    content = tf.readlines()
    for line in content:
        if('RMSE:' in line):    
            err = float(line.split('RMSE:')[1].replace('\n',''))
            val_errlist.append(err)


#%%
len(trainf_errlist), len(val_errlist), len(testf_errlist) 

#%%
y_train = [index for index in range(len(trainf_errlist))]
y_validation = [index*2 for index in range(len(val_errlist))]
y_testing = [index*2 for index in range(len(testf_errlist))]

#%%
fig = plt.figure(figsize=(6, 4))
plt.plot(y_train, trainf_errlist , label='Train loss')
plt.plot(y_validation, val_errlist , label='Val. loss')
plt.plot(y_testing, testf_errlist , label='Test loss')


plt.legend(loc='best', framealpha=0.5, prop={'size': 'large', 'family': 'monospace'})

# plt.show()

fig.savefig(R'{}/Loss.png'.format(directory), facecolor='w')
plt.clf()