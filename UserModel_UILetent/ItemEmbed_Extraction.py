#%%
from preprocessing import Preprocess
from model import  HANN
import tqdm
#%%
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

#%%
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
directory = '1126_clothing_lr5e07'

#%%
InterGRU = torch.load(R'ReviewsPrediction_Model/UserItemFeature/model/{}/InterGRU_epoch{}'.format(directory, 48))

#%%
itemEmbed = InterGRU.itemEmbedding
torch.save(itemEmbed, R'PretrainingEmb/item_embedding.pth')

# %%
print(InterGRU)

#%%
print(itemEmbed)

#%%
InterGRU2 = torch.load(R'ReviewsPrediction_Model/model/1127_clothing_catitem_lr5e07/InterGRU_epoch48')
print(InterGRU2)