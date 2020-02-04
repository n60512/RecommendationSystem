import torch
import torch.nn as nn
import torch.nn.functional as F

class IntraReviewGRU(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0, latentK = 64, method = 'dualFC'):
        super(IntraReviewGRU, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding
        self.method = method


        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            self.linear_alpha = torch.nn.Linear(hidden_size, 1) 

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'dualFC':
            self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear_alpha = torch.nn.Linear(hidden_size, 1)       

        self.intra_review = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), 
                          bidirectional=True)
                         
    def CalculateAttn(self, key_vector, query_vector):
        
        # Calculate weighting score
        if(self.method == 'dualFC'):
            x = F.relu(self.linear1(key_vector) +
                    self.linear2(query_vector) 
                )
            weighting_score = self.linear_alpha(x)
            # Calculate attention score
            intra_attn_score = torch.softmax(weighting_score, dim = 0)

        elif (self.method=='dot'):
            intra_attn_score = key_vector * query_vector
            
        elif (self.method=='general'):
            energy = self.attn(query_vector)
            x = F.relu(key_vector * energy)
            weighting_score = self.linear_alpha(x)
            # Calculate attention score            
            intra_attn_score = torch.softmax(weighting_score, dim = 0)


        return intra_attn_score
        
    def forward(self, input_seq, input_lengths, item_index, user_index, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)           
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        # Forward pass through GRU
        outputs, hidden = self.intra_review(packed, hidden)
 
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        # Calculate element-wise product
        elm_w_product = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # Calculate attention score

        if self.method != 'without':
            intra_attn_score = self.CalculateAttn(outputs, elm_w_product)
            new_outputs = intra_attn_score * outputs
            intra_outputs = torch.sum(new_outputs , dim = 0)    # output sum
        else:
            intra_outputs = torch.sum(outputs , dim = 0)    # output sum
            intra_attn_score = None

        # Return output and final hidden state
        return intra_outputs, hidden, intra_attn_score

#%% 
class HANN(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0, latentK = 64, isCatItemVec=False, method='dualFC'):
        super(HANN, self).__init__()
        
        self.method = method
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.latentK = latentK

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding

        self.isCatItemVec = isCatItemVec

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            self.linear_beta = torch.nn.Linear(hidden_size, 1)   
            
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'dualFC':            
            self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear4 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear_beta = torch.nn.Linear(hidden_size, 1)      

        if(self.isCatItemVec):
            GRU_InputSize = hidden_size*2
            # GRU_InputSize = hidden_size + latentK   # word dim. + item dim.
        else:
            GRU_InputSize = hidden_size

        self.inter_review = nn.GRU(GRU_InputSize, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
                          
        self.dropout = nn.Dropout(dropout)
        
        self.fc_doubleK = nn.Linear(hidden_size*2 , self.latentK*2)
        # self.fc_doubleK = nn.Linear(hidden_size + latentK , self.latentK*2)
        self.fc_singleK = nn.Linear(self.latentK*2, self.latentK)
        self.fc_out = nn.Linear(self.latentK, 1)
    
    def CalculateAttn(self, key_vector, query_vector):
        
        # Calculate weighting score
        if(self.method == 'dualFC'):
            x = F.relu(self.linear3(key_vector) +
                    self.linear4(query_vector) 
                )
            weighting_score = self.linear_beta(x)
            # Calculate attention score
            inter_attn_score = torch.softmax(weighting_score, dim = 0)

        elif (self.method=='dot'):
            inter_attn_score = key_vector * query_vector
            
        elif (self.method=='general'):
            energy = self.attn(query_vector)
            x = F.relu(key_vector * energy)
            weighting_score = self.linear_beta(x)
            # Calculate attention score            
            inter_attn_score = torch.softmax(weighting_score, dim = 0)


        return inter_attn_score

    def forward(self, intra_outputs, this_item_index, item_index, user_index, hidden=None):
        
        if(self.isCatItemVec):
            # Concat. intra output && item feature
            item_feature = self.itemEmbedding(this_item_index)
            inter_input = torch.cat((intra_outputs, item_feature), 2)
        else:
            inter_input = intra_outputs

        # Forward pass through GRU
        outputs, hidden = self.inter_review(inter_input, hidden)

        # Calculate element-wise product
        elm_w_product_inter = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # a = outputs * elm_w_product_inter
        # Calculate attention score
        inter_attn_score = self.CalculateAttn(outputs, elm_w_product_inter)

        # Consider attention score
        weighting_outputs = inter_attn_score * outputs
        outputs_sum = torch.sum(weighting_outputs , dim = 0)  

        # dropout
        # outputs_sum = self.dropout(outputs_sum)

        # Concat. interaction vector & GRU output
        outputs_cat = torch.cat((outputs_sum, elm_w_product_inter), dim=1)
        
        # hidden_size to 2*K dimension
        outputs_ = self.fc_doubleK(outputs_cat) 
        # 2*K to K dimension
        outputs_ = self.fc_singleK(outputs_)  
        # K to 1 dimension
        outputs_ = self.fc_out(outputs_)

        # dropout
        outputs_ = self.dropout(outputs_)

        sigmoid_outputs = torch.sigmoid(outputs_)
        sigmoid_outputs = sigmoid_outputs.squeeze(0)

        # Return output and final hidden state
        return sigmoid_outputs, hidden, inter_attn_score

# Decoder GRU
class DecoderGRU(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderGRU, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_step, last_hidden):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        rnn_output = torch.tanh(rnn_output)

        # Predict next word using Luong eq. 6
        output = self.out(rnn_output)
        # output = F.softmax(output, dim=1)

        # log softmax
        output = self.logsoftmax(output)

        # Return output and final hidden state
        return output, hidden