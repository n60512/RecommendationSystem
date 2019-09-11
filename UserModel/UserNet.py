#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import re
import os
import unicodedata
import codecs
import itertools

#%%
import math
import numpy as np
import time
from DBconnector import DBConnection

#%%
conn = DBConnection()
conn.connection
#%%
# sql = ('SELECT `reviewerID`,`asin` ' +
#     'FROM review ' +
#     'ORDER BY reviewerID ;'
#     )
sql = ('SELECT DISTINCT(`reviewerID`) ' +
    'FROM review ' +
    'ORDER BY reviewerID;'
    )
res = conn.selection(sql)
#%%
reviewerID = [res[i]['reviewerID'] for i in range(len(res))]
reviewerID[:10]
#%%
def WriteRelate(idlist, fileName = 'reviewerID'):
    with open('{}.csv'.format(fileName), 'a') as file:
        for value in idlist:
            file.write(value + ',')

#%%
