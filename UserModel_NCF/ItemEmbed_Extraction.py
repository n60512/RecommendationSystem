#%%
from preprocessing import Preprocess
from model import  NCF
import tqdm
#%%
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

#%%
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

#%%
model = torch.load(R'ReviewsPrediction_Model/NCF/model/1127_clothing_ncf_lr25e07/InterGRU_epoch22')

#%%
itemEmbed = model.itemEmbedding
torch.save(itemEmbed, R'PretrainingEmb/item_embedding_fromNCF.pth')

print(model)
print(itemEmbed)
