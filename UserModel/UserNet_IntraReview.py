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
class InterReview(nn.Module):
    def __init__(self, hidden_size, n_layers=1, dropout=0):
        super(InterReview, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = 64

        self.linear1 = torch.nn.Linear(hidden_size, 250)
        self.linear2 = torch.nn.Linear(hidden_size, 250)
        self.linear_alpha = torch.nn.Linear(250, 1)
        self.IntraReview()

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        
        self.attn = Attention(hidden_size)

        self.out = nn.Linear(hidden_size,self.output_size)
        self.out_ = nn.Linear(self.output_size,1)
#%% 
class IntraReview(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0):
        super(IntraReview, self).__init__()
        
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
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        
        self.attn = Attention(hidden_size)

        self.out = nn.Linear(hidden_size,self.output_size)
        self.out_ = nn.Linear(self.output_size,1)

    def forward(self, input_seq, input_lengths, item_index, user_index, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)        
        
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, current_hidden = self.gru(packed, hidden)
 
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        # Calculate element-wise product
        elm_w_product = self.attn(self.itemEmbedding(item_index), 
            self.userEmbedding(user_index))    

        # Calculate weighting score
        x = F.relu(self.linear1(outputs) +
                self.linear2(elm_w_product) 
            )
        weighting_score = self.linear_alpha(x)
        
        # Calculate attention score
        attn_score = torch.softmax(weighting_score)    

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
def batch2TrainData(myVoc, sentences, rating, this_asin, this_reviewerID, itemObj, userObj):
    sentences_rating_pair = list()
    for index in range(len(sentences)):
        sentences_rating_pair.append(
            [
                sentences[index], rating[index], 
                this_asin[index], this_reviewerID[index]
            ]
        )

    # Sort by sequence length
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
def loadData(datalen = 14000, batch_size = 5):

    print('Loading asin/reviewerID from cav file...')
    asin, reviewerID = Read_Asin_Reviewer()
    print('Loading asin/reviewerID complete.')

    # asin/reviewerID to index
    itemObj = item()
    itemObj.addItem(asin)
    userObj = user()
    userObj.addUser(reviewerID)

    print('Loading dataset from database...')

    sql = ('SELECT review.`ID`, review.reviewerID ' +
    ', review.`asin`, review.overall, review.reviewText ' +
    ', metadata.title ' +
    'FROM review ' +
    'LEFT JOIN metadata ' +
    'ON review.`asin` = metadata.`asin` ' +
    'LIMIT {};'.format(datalen)
    )

    conn = DBConnection()
    res = conn.selection(sql)
    conn.close()

    myVoc = Voc('Review')
    for row in res:
        current_sentence = row['reviewText']
        current_sentence = normalizeString(current_sentence)
        myVoc.addSentence(current_sentence)
    

    sentences = list()
    rating = list()

    # The asin and reviewerID which this review associate
    this_asin = list()
    this_reviewerID = list()

    for index in range(len(res)):
        sentences.append(
            normalizeString(
            res[index]['reviewText']
            )
        )

        rating.append(
            res[index]['overall']
        )

        this_asin.append(
            res[index]['asin']
        )

        this_reviewerID.append(
            res[index]['reviewerID']
        )

    training_batches = list()
    for index in range(0 , len(sentences)-1 , batch_size):
        # sentence_rating_pair = sentences[index:index+5], rating[index:index+5]
        training_batches.append( 
            batch2TrainData(
                myVoc, 
                sentences[index:index + batch_size], 
                rating[index:index + batch_size], 
                this_asin[index:index + batch_size], 
                this_reviewerID[index:index + batch_size],
                itemObj,
                userObj
                ) 
            )
    """
        training_batches -> 
        (input_variable, lengths, rating)
    """
    # print(training_batches[0])
    print('Loading complete.')

    return training_batches, myVoc, itemObj, userObj

#%% Train model
def train(training_batches, myVoc):
    # Configure models
    hidden_size = 300
    batch_size = 16

    # Get asin and reviewerID from file
    asin, reviewerID = Read_Asin_Reviewer()

    # Initialize textual embeddings
    embedding = nn.Embedding(myVoc.num_words, hidden_size)

    # Initialize asin/reviewer embeddings
    asin_embedding = nn.Embedding(len(asin), hidden_size)
    reviewerID_embedding = nn.Embedding(len(reviewerID), hidden_size)

    # Construct item lookup table
    itemObj = item()
    itemObj.addItem(asin)

    # Construct user lookup table
    userObj = user()
    userObj.addUser(reviewerID)

    # Initialize encoder & decoder models
    rnn = IntraReview(hidden_size, embedding, asin_embedding, reviewerID_embedding)
    # Use appropriate device
    rnn = rnn.to(device)
    print('Models built and ready to go!')

    # Configure training/optimization
    learning_rate = 0.0000001
    rnn.train()

    # Initialize optimizers
    print('Building optimizers ...')
    rnn_optimizer = optim.Adam(rnn.parameters(), 
        lr=learning_rate, weight_decay=0.001)

    # Zero gradients
    rnn_optimizer.zero_grad()

    criterion = nn.MSELoss()
    loss = 0
    current_loss = 0 
    all_losses = []
    store_every = 10

    for Epoch in range(100):
        for iteration in range(len(training_batches)):
            training_batch = training_batches[iteration]
            
            # input_variable = training_batch[0]
            # lengths = training_batch[1]
            input_variable, lengths, rating , asin_index, reviewerID_index = training_batch

            # Set device options
            input_variable = input_variable.to(device)
            lengths = lengths.to(device)
            asin_index = asin_index.to(device)
            reviewerID_index = reviewerID_index.to(device)

            normalize_rating = (rating - 1)/ (5-1)
            normalize_rating = normalize_rating.to(device)

            # Forward pass through encoder
            rnn_outputs, rnn_hidden, attn_score = rnn(input_variable, lengths, 
                asin_index, reviewerID_index)
                        
            # loss = criterion(rnn_outputs.squeeze(1), normalize_rating)
            err = rnn_outputs.squeeze(1) - normalize_rating
            loss = torch.sum(torch.mul(err, err) , dim = 0)

            loss.backward()
            current_loss += loss

            rnn_optimizer.step()
        
        all_losses.append(current_loss / len(training_batches))
        current_loss = 0

        print('Epoch:{}\tLoss:{}'.format(Epoch, all_losses[Epoch]))

        if Epoch % store_every == 0:
            torch.save(rnn, R'ReviewsPrediction_Model\ReviewsPrediction_{}'.format(Epoch))

#%%
class Evaluate(nn.Module):
    def __init__(self, rnn):
        super(Evaluate, self).__init__()
        self.rnn = rnn

    def forward(self, input_batch, lengths, asin_index, reviewerID_index):
        rnn_outputs , rnn_hidden, attn_score = self.rnn(input_batch, lengths, 
            asin_index, reviewerID_index)
        return rnn_outputs, rnn_hidden , attn_score

#%%
def evaluateData(datalen = 50):

    sql = ('SELECT review.`ID`, review.reviewerID ' +
    ', review.`asin`, review.overall, review.reviewText ' +
    ', metadata.title ' +
    'FROM review ' +
    'LEFT JOIN metadata ' +
    'ON review.`asin` = metadata.`asin` ' +
    'WHERE review.`ID`>15000 ' +
    'LIMIT {};'.format(datalen)
    )

    conn = DBConnection()
    return conn.selection(sql)
#%%
def evaluate(rnn , voc , itemObj, userObj):

    evaluator = Evaluate(rnn)
    data = evaluateData(3000)

    sentence_attn_score_ls = list()
    current_loss = 0
    counter = 0

    for row in data:
        
        true_rating = row['overall']
        sentence = row['reviewText']
        itemid = row['asin']
        userid = row['reviewerID']
        
        # Get index
        asin_index = torch.tensor([
            itemObj.asin2index[itemid]
        ])
        reviewerID_index = torch.tensor([
            userObj.reviewerID2index[userid]
        ])        

        # Normalize String
        sentence = normalizeString(sentence)
        # words -> indexes
        indexes_batch = [indexesFromSentence_Evaluate(voc, sentence)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        asin_index = asin_index.to(device)
        reviewerID_index = reviewerID_index.to(device)        

        # Evaluate
        output, hidden, attn_score = evaluator(input_batch, lengths, 
            asin_index, reviewerID_index)

        # Attention record
        counter = 0
        sentence_attn_score = dict()
        seg_sentence = sentence.split(' ')
        for val in seg_sentence[:199]:
            try:
                sentence_attn_score[seg_sentence[counter]] = attn_score[counter].item()
                pass
            except IndexError as msg:
                print(len(seg_sentence))
                print(len(attn_score))
                break
                pass            
            counter += 1

        sentence_attn_score_ls.append(sentence_attn_score)
        
        # Calculate loss
        predict_rating = (output*(5-1))+1        
        normalize_rating = (true_rating - 1)/ (5-1)
        current_loss += (normalize_rating - output)**2
        counter += 1

        if(False):
            print('==================================')
            print('True :{}\t Predict :{}\t closs :{}'.format(
                    normalize_rating, output.item(), 
                    (normalize_rating - output).item())
                )

        pass


    RMSE = math.sqrt(
        current_loss/counter
    )
    
    print('==================================')
    print('RMSE :{}\t'.format(RMSE))

    return sentence_attn_score_ls

#%%
if __name__ == "__main__":

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    
    # Load in training batches
    batch_size = 14000
    training_batches, myVoc, itemObj, userObj = loadData(batch_size)

    if(not True):
        train(training_batches , myVoc)
    
    if(True):
        for index in range(0, 100, 10):
            rnn = torch.load(R'ReviewsPrediction_Model\ReviewsPrediction_{}'.format(index))
            evaluate(rnn, myVoc, itemObj, userObj)

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

#%%
jupyter = False
if(True):
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
        
    # Load in training batches
    batch_size = 14000
    training_batches, myVoc, itemObj, userObj = loadData(batch_size)

    rnn = torch.load(R'ReviewsPrediction_Model\ReviewsPrediction_{}'.
        format(90))
    attn_weight = evaluate(rnn, myVoc, itemObj, userObj)

    writejs = False
    if(writejs):
        for index, attn in enumerate(attn_weight):
            js(index, attn)



#%%

import seaborn as sns 
import pandas as pd
from matplotlib.collections import QuadMesh
import matplotlib.pyplot as plt

def show_weifig(attn_weight, fig_x = 12, fig_y = 7):
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    sns.set(font=['SimHei'], font_scale=1.5)
    sns.set_style('whitegrid',{'font.sans-serif':['SimHei']})

    fig, ax1 = plt.subplots()
    fig.set_size_inches(90, 50)

    weight = np.fromiter(attn_weight.values(), dtype=float)
    weight = np.expand_dims(weight, axis=0)
    
    word = [list(attn_weight.keys())]
    word = np.asarray(word)

    # fig = sns.heatmap(word, annot=True, ax=ax1)
    fig = sns.heatmap(weight, annot=word, cbar=True, linewidths=0.2, square=False, cmap="YlGnBu", fmt = '')
    
    # plt.show()

    figure = fig.get_figure()    
    figure.savefig('test.png', facecolor='w')
    plt.clf()