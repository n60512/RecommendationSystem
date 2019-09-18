#%%
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

#%%
with open('asin.csv','r') as file:
    content = file.read()
    asin = content.split(',')
    print('asin count : {}'.format(len(asin)))

with open('reviewerID.csv','r') as file:
    content = file.read()
    reviewerID = content.split(',')
    print('reviewerID count : {}'.format(len(reviewerID)))

#%%
class item:
    def __init__(self):
        self.asin2index = {}
        self.index2asin = {}
        self.num_asins = 0

    def addItem(self, asin):
        for id_ in asin:
            self.asin2index[id_] = self.num_asins
            self.index2asin[self.num_asins] = id_
            self.num_asins += 1
    pass

class user:
    def __init__(self):
        self.reviewerID2index = {}
        self.index2reviewerID = {}
        self.num_reviewerIDs = 0

    def addUser(self, reviewerID):
        for id_ in reviewerID:
            self.reviewerID2index[id_] = self.num_reviewerIDs
            self.index2reviewerID[self.num_reviewerIDs] = id_
            self.num_reviewerIDs += 1   
    pass

#%%
hidden_size = 500
asin_embedding = nn.Embedding(len(asin), hidden_size)
reviewerID_embedding = nn.Embedding(len(reviewerID), hidden_size)

#%%
asin_embedding, reviewerID_embedding

#%%
itemObj = item()
itemObj.addItem(asin)

#%%
userObj = user()
userObj.addUser(reviewerID)

#%%
print('Asin [B00KC7I2GU] index :{}'.format(itemObj.asin2index['B00KC7I2GU']))
print('itemObj len :{}'.format(len(itemObj.asin2index)))

#%%
print('ReviewerID [AZWOAIK9NREG3] index :{}'.format(userObj.reviewerID2index['AZWOAIK9NREG3']))
print('userObj len :{}'.format(len(userObj.reviewerID2index)))


#%% element-wise product practice

# Get index
asin_ts = torch.LongTensor([itemObj.asin2index['B00KC7I2GU']])
reviewerID_ts = torch.LongTensor([userObj.reviewerID2index['AZWOAIK9NREG3']])

# Index to embedding
asin_embedding = asin_embedding(asin_ts)
reviewerID_embedding = reviewerID_embedding(reviewerID_ts)

#%%
asin_embedding.size(), reviewerID_embedding.size()

#%%
element_wise_product = asin_embedding * reviewerID_embedding
element_wise_product.size()

#%%
test = asin_embedding.mm(reviewerID_embedding.transpose(0, 1))
test.size()

#%%
