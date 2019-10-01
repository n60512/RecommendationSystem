#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
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
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size        
    
    def element_wise_product(self, itemVec, userVec):
        return itemVec * userVec

    def forward(self, itemVec, userVec):
        return self.element_wise_product(itemVec, userVec)
    pass

#%% 
class HANN(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0):
        super(HANN, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = 64

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding

        self.linear1 = torch.nn.Linear(hidden_size, 250)
        self.linear2 = torch.nn.Linear(hidden_size, 250)
        self.linear_alpha = torch.nn.Linear(250, 1)

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.intra_review = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), 
                          bidirectional=True)
        
        self.intra_attn = Attention(hidden_size)

        self.linear3 = torch.nn.Linear(hidden_size, 250)
        self.linear4 = torch.nn.Linear(hidden_size, 250)
        self.linear_beta = torch.nn.Linear(250, 1)

        self.inter_review = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), 
                          bidirectional=True)
        
        self.inter_attn = Attention(hidden_size)        

        self.out = nn.Linear(hidden_size,self.output_size)
        self.out_ = nn.Linear(self.output_size,1)

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
        weighting_score = self.linear_alpha(x)
        
        # Calculate attention score
        attn_score = torch.softmax(weighting_score, dim = 0)    

        new_outputs = attn_score * outputs
        # new_outputs = torch.sum(new_outputs , dim = 0)    # output sum

        intra_outputs = new_outputs

        # Forward pass through GRU
        outputs, current_hidden = self.inter_review(intra_outputs, hidden)
 
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        # Calculate element-wise product
        elm_w_product_inter = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # Calculate weighting score
        x = F.relu(self.linear1(outputs) +
                self.linear2(elm_w_product_inter) 
            )
        weighting_score = self.linear_beta(x)
        
        # Calculate attention score
        attn_score = torch.softmax(weighting_score, dim = 0)    

        new_outputs = attn_score * outputs
        new_outputs_sum = torch.sum(new_outputs , dim = 0)    

        # hidden_size to 64 dimension
        new_outputs = self.out(new_outputs_sum)  
        # 64 to 1 dimension
        new_outputs = self.out_(new_outputs)    
        sigmoid_outputs = torch.sigmoid(new_outputs)

        if(False):
            print('input_seq size:\n',input_seq.size())
            print('embedded size:\n',embedded.size())
            print('Output size:\n',outputs.size())
            print('elm_w_product:\n',elm_w_product.size())
            print(' x:\n',x.size())
            print(' weighting_score:\n',weighting_score.size())
            print(' attn_score:\n',attn_score.size())
            print(' new_outputs:\n',new_outputs.size())
            print('new_outputs_sum size:\n',new_outputs_sum.size())
            stop =1

        # Return output and final hidden state
        return sigmoid_outputs, current_hidden, attn_score
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
PAD_token = 0  # Used for padding short sentences
class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD"}
        self.num_words = 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
    pass

#%% convert all letters to lowercase 
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def indexesFromSentence(voc, sentence , MAX_LENGTH = 200):
    sentence_segment = sentence.split(' ')[:MAX_LENGTH]
    return [voc.word2index[word] for word in sentence_segment]

def indexesFromSentence_Evaluate(voc, sentence , MAX_LENGTH = 200):
    sentence_segment = sentence.split(' ')[:MAX_LENGTH]
    indexes = list()
    for word in sentence_segment:
        try:
            indexes.append(voc.word2index[word])
        except KeyError as ke:
            indexes.append(PAD_token)
        except Exception as msg:
            print('Exception :\n', msg)

    return indexes

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths
                
# Returns all items for a given batch of pairs
def batch2TrainData(myVoc, sentences, rating, this_asin, this_reviewerID, itemObj, userObj, isSort = False):
    sentences_rating_pair = list()
    for index in range(len(sentences)):
        sentences_rating_pair.append(
            [
                sentences[index], rating[index], 
                this_asin[index], this_reviewerID[index]
            ]
        )

    # Sort by sequence length
    if(isSort):
        sentences_rating_pair.sort(key=lambda x: len(x[0].split(" ")), reverse=True)

    sentences_batch, rating_batch, this_asin_batch, this_reviewerID_batch  = [], [], [], []
    for pair in sentences_rating_pair:
        sentences_batch.append(pair[0])
        rating_batch.append(pair[1])
        this_asin_batch.append(itemObj.asin2index[pair[2]])
        this_reviewerID_batch.append(userObj.reviewerID2index[pair[3]])
        
        ## debug
        if(False):
            print('asin:{}\t index:{}'.format(pair[2], itemObj.asin2index[pair[2]]))
        # print('reviewerID:{}\t index:{}'.format(pair[3], userObj.reviewerID2index[pair[3]]))

    inp, lengths = inputVar(
        sentences_batch,
        myVoc)
    
    label = torch.tensor([val for val in rating_batch])
    # asin and reviewerID batch association
    asin_batch = torch.tensor([val for val in this_asin_batch])
    reviewerID_batch = torch.tensor([val for val in this_reviewerID_batch])

    return inp, lengths, label, asin_batch, reviewerID_batch

#%%
def Read_Asin_Reviewer():
    with open('asin.csv','r') as file:
        content = file.read()
        asin = content.split(',')
        print('asin count : {}'.format(len(asin)))

    with open('reviewerID.csv','r') as file:
        content = file.read()
        reviewerID = content.split(',')
        print('reviewerID count : {}'.format(len(reviewerID)))
    
    return asin, reviewerID

#%% Load dataset from database
def loadData():

    print('Loading asin/reviewerID from cav file...')
    asin, reviewerID = Read_Asin_Reviewer()
    print('Loading asin/reviewerID complete.')

    # asin/reviewerID to index
    itemObj = item()
    itemObj.addItem(asin)
    userObj = user()
    userObj.addUser(reviewerID)

    st = time.time()
    print('Loading dataset from database...') 

    sql = (
        'WITH tenReviewsUp AS ( ' +
        '		SELECT reviewerID ' +
        '		FROM review ' +
        '		group by reviewerID ' +
        '		HAVING COUNT(reviewerID) >= 10 ' +
        '	) ' +
        'SELECT ' +
        'RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, ' +
        'review.`ID`, review.reviewerID , review.`asin`, review.overall, review.reviewText, review.unixReviewTime ' +
        'FROM review , metadata ' +
        'WHERE reviewerID IN (SELECT * FROM tenReviewsUp) ' +
        'AND review.`asin` = metadata.`asin` ' +
        'ORDER BY reviewerID,unixReviewTime ASC ;' 
    )

    conn = DBConnection()
    res = conn.selection(sql)
    conn.close()

    print('Loading complete. [{}]'.format(time.time()-st))
    print('Creating Voc ...') 

    st = time.time()
    # Creating Voc
    myVoc = Voc('Review')
    for row in res:
        if(row['rank'] < 11):
            current_sentence = row['reviewText']
            current_sentence = normalizeString(current_sentence)
            myVoc.addSentence(current_sentence)
        else:
            this_is_testing_data = 1

    print('Voc creation complete. [{}]'.format(time.time()-st))
    
    return res, myVoc, itemObj, userObj

#%%
class PersonalHistory:
    def __init__(self, reviewerID):
        self.reviewerID = reviewerID
        self.sentences = list()
        self.rating = list()
        self.this_asin = list()
        self.this_reviewerID = list()
        self.RowCount = 0

    def addData(self, sentence_, rating_, this_asin_, reviewerID_):
        self.sentences.append(sentence_)
        self.rating.append(rating_)
        self.this_asin.append(this_asin_)
        self.this_reviewerID.append(reviewerID_)
        self.RowCount += 1
        pass
    pass

#%% 根據使用者的回復數決定 batch
def GenerateBatches(res, itemObj, userObj, batch_size = 7):
    
    USER = list()
    LAST_USER = ''
    ctr = -1

    st = time.time()
    print('Generating batches ...') 

    for index in range(len(res)):
        # Check is the next user or not
        if(LAST_USER != res[index]['reviewerID']):
            LAST_USER = res[index]['reviewerID']
            USER.append(PersonalHistory(res[index]['reviewerID']))
            ctr += 1   # add counter if change the user id
            
        USER[ctr].addData(
                    normalizeString(res[index]['reviewText']),
                    res[index]['overall'], 
                    res[index]['asin'],
                    res[index]['reviewerID']
                )

    training_batches = dict()
    for index in range(len(USER)):
        id_ = USER[index].reviewerID
        training_batches[id_] = batch2TrainData(
                                    myVoc, 
                                    USER[index].sentences[:batch_size], 
                                    USER[index].rating[:batch_size], 
                                    USER[index].this_asin[:batch_size], 
                                    USER[index].this_reviewerID[:batch_size],
                                    itemObj,
                                    userObj
                                )

    if(False):
        print('[{}] : '.format( 
            USER[2].reviewerID
            ))
        for sen in USER[2].sentences:
            print(len(sen.split(' ')), end = ',')
        print(training_batches[ USER[2].reviewerID ])
    
    print('User length :{}'.format(ctr))
    print('Batch generation complete . [{}]'.format(time.time()-st)) 
            
    return training_batches

#%% Train model
def train(training_batches, myVoc):
    # Configure models
    hidden_size = 300

    # Get asin and reviewerID from file
    asin, reviewerID = Read_Asin_Reviewer()

    # Initialize textual embeddings
    embedding = nn.Embedding(myVoc.num_words, hidden_size)

    # Initialize asin/reviewer embeddings
    asin_embedding = nn.Embedding(len(asin), hidden_size)
    reviewerID_embedding = nn.Embedding(len(reviewerID), hidden_size)

    # Initialize encoder & decoder models
    IntraGRU = HANN(hidden_size, embedding, asin_embedding, reviewerID_embedding)
    # Use appropriate device
    IntraGRU = IntraGRU.to(device)
    print('Models built and ready to go!')

    # Configure training/optimization
    learning_rate = 0.0000001
    IntraGRU.train()

    # Initialize optimizers
    print('Building optimizers ...')
    IntraGRU_optimizer = optim.Adam(IntraGRU.parameters(), 
        lr=learning_rate, weight_decay=0.001)
    
    # Zero gradients
    IntraGRU_optimizer.zero_grad()

    loss = 0
    current_loss = 0 
    all_losses = []
    store_every = 10

    for Epoch in range(101):
        iteration = 0
        for key_userid, training_batch in training_batches.items():
        # for iteration in range(len(training_batches)):
        #     training_batch = training_batches[iteration]
            
            input_variable, lengths, rating , asin_index, reviewerID_index = training_batch

            # Set device options
            input_variable = input_variable.to(device)
            lengths = lengths.to(device)
            asin_index = asin_index.to(device)
            reviewerID_index = reviewerID_index.to(device)

            # Forward pass through encoder
            outputs, intra_hidden, attn_score = IntraGRU(input_variable, lengths, 
                asin_index, reviewerID_index)


            # Calculate Loss
            normalize_rating = (rating - 1)/ (5-1)
            normalize_rating = normalize_rating.to(device)

            err = outputs.squeeze(1) - normalize_rating
            loss = torch.sum(torch.mul(err, err) , dim = 0)

            loss.backward()
            current_loss += loss

            IntraGRU_optimizer.step()

            iteration += 1
        
        all_losses.append(current_loss / len(training_batches))
        current_loss = 0

        print('Epoch:{}\tLoss:{}'.format(Epoch, all_losses[Epoch]))

        if Epoch % store_every == 0:
            torch.save(IntraGRU, R'ReviewsPrediction_Model\ReviewsPrediction_{}'.format(Epoch))

#%%
if __name__ == "__main__":

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    
    # Load in training batches
    res, myVoc, itemObj, userObj = loadData()

    if(False):
        ctr_ = 0
        for key, val in training_batches.items():
            print('\n=======================\n{}\t{}'.
                format(key, userObj.reviewerID2index[key])
                )
            print(val[4])
            ctr_ += 1
            if(ctr_>10):
                break        
    
    if(True):
        training_batches = GenerateBatches(
            res, 
            itemObj, 
            userObj,
            batch_size = 7
        )
        train(training_batches, myVoc)


    pass


# tensorboard --logdir=data/ --host localhost --port 8088

#%%
def js(index, attn_weight):
    with open(R'AttentionVisualize\{}.html'.format(index),'a') as file:
        text = "<!DOCTYPE html>\n<html>\n<body>\n<head>\n<script src='https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js'></script>\n</head><h1>Attn</h1><div id='text'>text goes here</div>\n"
        file.write(text)
        file.write('\n<script>\nvar words = [')

    for key, val in attn_weight.items():
        with open(R'AttentionVisualize\{}.html'.format(index),'a') as file:
            file.write("{{\n'word': '{}',".format(key))
            file.write("'attention': {},\n}},".format(val))
    
    with open(R'AttentionVisualize\{}.html'.format(index),'a') as file:
        file.write("];\n$('#text').html($.map(words, function(w) {\nreturn '<span style=\"background-color:hsl(360,100%,' + (w.attention * -50 + 100) + '%)\">' + w.word + ' </span>'\n}))\n</script>")
