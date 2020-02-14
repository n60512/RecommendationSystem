
import tqdm
import torch
import torch.nn as nn
from torch import optim
import random


model = torch.load(R'HNAE/log/origin/20200116_09_38_interaction@6_review@5_nltk_fasttext_rand42/Model/IntraGRU_idx0_epoch0')
print("Intra: \n{}\n".format(model))

# Loading InterGRU
InterGRU = torch.load(R'HNAE/log/origin/20200116_09_38_interaction@6_review@5_nltk_fasttext_rand42/Model/InterGRU_epoch0')
print("Intra: \n{}\n".format(InterGRU))