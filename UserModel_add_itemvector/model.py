from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

class IntraReviewGRU(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0):
        super(IntraReviewGRU, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding

        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_alpha = torch.nn.Linear(hidden_size, 1)

        self.intra_review = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=dropout, 
                          bidirectional=True)
                         

    def forward(self, input_seq, input_lengths, item_index, user_index, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)        
        
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        # Forward pass through GRU
        outputs, current_hidden = self.intra_review(packed, hidden)
 
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        # Calculate element-wise product
        elm_w_product = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # Calculate weighting score
        x = F.relu(self.linear1(outputs) +
                self.linear2(elm_w_product)
            )
        weighting_score = self.linear_alpha(x)  # Constant
        
        # Calculate attention score
        intra_attn_score = torch.softmax(weighting_score, dim = 0)    

        new_outputs = intra_attn_score * outputs
        intra_outputs = torch.sum(new_outputs , dim = 0)    # output sum

        # intra_outputs = intra_outputs.unsqueeze(dim=1)

        # Return output and final hidden state
        return intra_outputs, current_hidden, intra_attn_score

#%% 
class HANN(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0, latentK = 64, isCatItemVec=False):
        super(HANN, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.latentK = latentK

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding

        self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear4 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_beta = torch.nn.Linear(hidden_size, 1)
        self.isCatItemVec = isCatItemVec

        if(self.isCatItemVec):
            GRU_InputSize = hidden_size*2
        else:
            GRU_InputSize = hidden_size

        self.inter_review = nn.GRU(GRU_InputSize, hidden_size, n_layers,
                          dropout=dropout)
        
        self.out128 = nn.Linear(hidden_size*2 , self.latentK*2)
        self.out64 = nn.Linear(self.latentK*2, self.latentK)
        self.out_ = nn.Linear(self.latentK, 1)

    def forward(self, intra_outputs, this_item_index, item_index, user_index, hidden=None):
        
        if(self.isCatItemVec):
            # Concat. intra output && item feature
            item_feature = self.itemEmbedding(this_item_index)
            inter_input = torch.cat((intra_outputs, item_feature), 2)
        else:
            inter_input = intra_outputs

        # Forward pass through GRU
        outputs, current_hidden = self.inter_review(inter_input, hidden)

        # Calculate element-wise product
        elm_w_product_inter = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        test_outputs = outputs.squeeze(dim=1)
        # Calculate weighting score
        x = F.relu(self.linear3(test_outputs) +
                self.linear4(elm_w_product_inter) 
            )
        weighting_score = self.linear_beta(x)
            
        # Calculate attention score (size: [200,15,1])
        inter_attn_score = torch.softmax(weighting_score, dim = 0)
        
        # Consider attention score
        new_outputs = inter_attn_score * outputs
        new_outputs_sum = torch.sum(new_outputs , dim = 0)    

        # Concat. interaction vector & GRU output
        new_outputs_cat = torch.cat((new_outputs_sum, elm_w_product_inter) ,1)

        # hidden_size to 128 dimension
        new_outputs = self.out128(new_outputs_cat) 
        # hidden_size to 64 dimension
        new_outputs = self.out64(new_outputs)  
        # 64 to 1 dimension
        new_outputs = self.out_(new_outputs)    
        sigmoid_outputs = torch.sigmoid(new_outputs)
        sigmoid_outputs = sigmoid_outputs.squeeze(0)

        # Return output and final hidden state
        return sigmoid_outputs, current_hidden, inter_attn_score