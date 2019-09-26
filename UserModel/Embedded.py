#%%
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
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

#%%
torch.tensor([1,2,3,4,5])
#%% element-wise product practice

# Get index
asin_ts = torch.LongTensor([itemObj.asin2index['B00KC7I2GU']])
reviewerID_ts = torch.LongTensor([userObj.reviewerID2index['AZWOAIK9NREG3']])

# Index to embedding
asin_ = asin_embedding(torch.tensor([1,2,3,4,5]))
reviewerID_ = reviewerID_embedding(torch.tensor([1,2,3,4,5]))

#%%
# asin_embedding(torch.tensor([1,2,3,4,5])).size()
#%%
asin_.size(), reviewerID_.size()

#%%
element_wise_product = asin_ * reviewerID_
element_wise_product.size()

#%%
test = asin_embedding.mm(reviewerID_embedding.transpose(0, 1))
test.size()

#%%
weight_hidden = Parameter(torch.Tensor(250, hidden_size))
weight_hidden.size()

#%%
a = torch.randn(201, 5,500 , dtype=torch.float)
print(a.size())

#%%
c = a.matmul(weight_hidden.transpose(0, 1))

#%%
c.size()

#%%
outputs = torch.randn(201,5,500 , dtype=torch.float)
weight_hidden = torch.randn(250,500 , dtype=torch.float)
weight_feature = torch.randn(250,500 , dtype=torch.float)
weight_alpha = torch.randn(1,250 , dtype=torch.float)

#%%
b = elm_w_product.matmul(weight_feature.transpose(0, 1))
b.size()

#%%
x = F.relu(outputs.matmul(weight_hidden.transpose(0, 1)) +
        elm_w_product.matmul(weight_feature.transpose(0, 1))
    ) 
        
print(' x:\n',x.size())

weighting_score = x.matmul(weight_alpha.transpose(0, 1))
#%%
x.size(), weighting_score.size()
#%%
for index in x[0][0]:
    print(index)

# x[0][0][0]

#%%
for index in weighting_score:
    print(index)

#%%
attn_score = torch.sigmoid(weighting_score)

#%%
attn_score.size()

#%%
for index in attn_score:
    print(index)

#%%
test = torch.tensor([[[.1,.1,.2,.4,.9],[.1,.1,.2,.4,.9]],
            [[.1,.1,.2,.4,.9],[.1,.1,.2,.4,.9]]])
test = torch.sigmoid(test)
test , test.size()

#%%
a = torch.randn(201,5,500 , dtype=torch.float)
b = torch.randn(201,5,1 , dtype=torch.float)
c = a * b
a[0][0][0], b[0][0][0], c[0][0][0]


#%%
a[0][0][2], b[0][0][0], c[0][0][2]

#%%
a[0][0][3], b[0][0][0], c[0][0][3]

#%%
x = torch.tensor([[0.0015],
        [0.4772],
        [0.9599],
        [0.9788],
        [0.0529]])
x = x.squeeze(1)
x
#%%
x.size()

#%%
a = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6])
b = torch.tensor([0.3, 0.4, 0.5, 0.5, 0.6])

torch.mul(a-b,a-b), 
torch.sum(torch.mul(a-b,a-b) , dim = 0) 

#%%
a = torch.tensor([[
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6]
    ],
    [
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6]
    ],
    [
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6]
    ],
    [
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6]
    ]]
    )

b = torch.tensor([
        [0.1, 0.2, 0.3, 0.8, 0.9],
    ]
    )

a.size(),b.size()

#%%
c = a.matmul(b.transpose(0, 1))
a.size(),b.size(),c.size()
#%%
c


#%%
d = Parameter(torch.Tensor().new_zeros((250, 300)))
d.size()


#%%
