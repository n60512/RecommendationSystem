from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

#%% 
class HANN(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0, latentK = 64):
        super(HANN, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.latentK = latentK

        self.itemEmbedding = itemEmbedding

        self.inter_review = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=dropout)
                          
        
        self.out128 = nn.Linear(hidden_size , self.latentK*2)
        self.out64 = nn.Linear(self.latentK*2, self.latentK)
        self.out_ = nn.Linear(self.latentK, 1)

    def forward(self, input_asin,  hidden=None):
        # Convert asin indexes to embeddings
        embedded = self.itemEmbedding(input_asin)

        # Forward pass through GRU
        outputs, current_hidden = self.inter_review(embedded, hidden)
        new_outputs = torch.sum(outputs , dim = 0)

        # hidden_size to 128 dimension
        new_outputs = self.out128(new_outputs) 
        # hidden_size to 64 dimension
        new_outputs = self.out64(new_outputs)  
        # 64 to 1 dimension
        new_outputs = self.out_(new_outputs)    
        sigmoid_outputs = torch.sigmoid(new_outputs)
        sigmoid_outputs = sigmoid_outputs.squeeze(0)

        # Return output and final hidden state
        return sigmoid_outputs, current_hidden