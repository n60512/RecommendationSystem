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
directory = '1130_clothing_bs16_lr1e05'

#%%
InterGRU = torch.load(R'ReviewsPrediction_Model/UserItemFeature/model/{}/InterGRU_epoch{}'.format(directory, 6))

#%%
itemEmbed = InterGRU.itemEmbedding
torch.save(itemEmbed, R'PretrainingEmb/item_embedding_fromGRU.pth')

