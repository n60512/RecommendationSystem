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
import tqdm

#%%
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
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0, latentK = 64):
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

        self.inter_review = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=dropout)
                          
        
        self.out128 = nn.Linear(hidden_size*2 , self.latentK*2)
        self.out64 = nn.Linear(self.latentK*2, self.latentK)
        self.out_ = nn.Linear(self.latentK, 1)

    def forward(self, intra_outputs, item_index, user_index, hidden=None):

        # Forward pass through GRU
        outputs, current_hidden = self.inter_review(intra_outputs, hidden)

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
        inter_attn_score = inter_attn_score.unsqueeze(dim=1)
            
        new_outputs = inter_attn_score * outputs
        new_outputs_sum = torch.sum(new_outputs , dim = 0)    
        
        ## tmp , no att
        new_outputs_sum = torch.sum(outputs , dim = 0)    

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
def inputVar(l, voc, testing=False):
    if(testing):
        indexes_batch = [indexesFromSentence_Evaluate(voc, sentence) for sentence in l]
    else:   # for training
        indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

#%%
def Read_Asin_Reviewer(talbe=''):
    with open('{}asin.csv'.format(talbe),'r') as file:
        content = file.read()
        asin = content.split(',')
        print('asin count : {}'.format(len(asin)))

    with open('{}reviewerID.csv'.format(talbe),'r') as file:
        content = file.read()
        reviewerID = content.split(',')
        print('reviewerID count : {}'.format(len(reviewerID)))
    
    return asin, reviewerID

#%% Load dataset from database
def loadData(havingCount = 15, LIMIT=5000, testing=False, table=''):

    print('Loading asin/reviewerID from cav file...')
    asin, reviewerID = Read_Asin_Reviewer(table)
    print('Loading asin/reviewerID complete.')

    # asin/reviewerID to index
    itemObj = item()
    itemObj.addItem(asin)
    userObj = user()
    userObj.addUser(reviewerID)

    st = time.time()
    print('Loading dataset from database...') 

    if(not testing):
        # load training user data
        sql = (
            'WITH tenReviewsUp AS ( ' +
            '		SELECT reviewerID ' +
            '		FROM {}review '.format(table) +
            '		group by reviewerID ' +
            '		HAVING COUNT(reviewerID) >= {} '.format(havingCount) +
            '		limit {} '.format(LIMIT) +
            '	) ' +
            'SELECT ' +
            'RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, ' +
            '{}review.`ID`, {}review.reviewerID , {}review.`asin`, {}review.overall, {}review.reviewText, {}review.unixReviewTime '.format(table, table, table, table, table, table) +
            'FROM {}review , {}metadata '.format(table, table) +
            'WHERE reviewerID IN (SELECT * FROM tenReviewsUp) ' +
            'AND {}review.`asin` = {}metadata.`asin` '.format(table, table) +
            'ORDER BY reviewerID,unixReviewTime ASC ;'
        )
    else:
        sql = (
            'WITH tenReviewsUp AS ( ' +
            '		SELECT reviewerID ' +
            '		FROM {}review '.format(table) +
            'WHERE reviewerID  NOT IN ' +
            '( ' +
            '   SELECT reviewerID ' +
            '   FROM toys_training_1000 ' +
            ') ' +
            '		group by reviewerID ' +
            '		HAVING COUNT(reviewerID) >= {} '.format(havingCount) +
            '		limit {} '.format(LIMIT) +
            '	) ' +
            'SELECT ' +
            'RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, ' +
            '{}review.`ID`, {}review.reviewerID , {}review.`asin`, {}review.overall, {}review.reviewText, {}review.unixReviewTime '.format(table, table, table, table, table, table) +
            'FROM {}review , {}metadata '.format(table, table) +
            'WHERE reviewerID IN (SELECT * FROM tenReviewsUp) ' +
            'AND {}review.`asin` = {}metadata.`asin` '.format(table, table) +
            'ORDER BY reviewerID,unixReviewTime ASC ;'
        )

    print(sql)

    conn = DBConnection()
    res = conn.selection(sql)
    conn.close()

    print('Loading complete. [{}]'.format(time.time()-st))
    
    return res, itemObj, userObj

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

# %%
def Generate_Voc_User(res, batch_size = 7, validation_batch_size=3, generateVoc=True):
    
    USER = list()
    LAST_USER = ''
    ctr = -1
    havingCount = batch_size + validation_batch_size

    # Creating Voc
    st = time.time()
    print('Creating Voc ...') 
    if(generateVoc):
        myVoc = Voc('Review')

    for index in tqdm.tqdm(range(len(res))):
        # Check is the next user or not
        if(LAST_USER != res[index]['reviewerID']):
            LAST_USER = res[index]['reviewerID']
            USER.append(PersonalHistory(res[index]['reviewerID']))
            ctr += 1   # add counter if change the user id
        
        if(res[index]['rank'] < havingCount + 1):
            current_sentence = normalizeString(res[index]['reviewText'])
            if(generateVoc):
                myVoc.addSentence(current_sentence) # myVoc add word !
            USER[ctr].addData(
                        current_sentence,
                        res[index]['overall'], 
                        res[index]['asin'],
                        res[index]['reviewerID']
                    )

    print('Voc creation complete. [{}]'.format(time.time()-st))

    if(generateVoc):
        return myVoc, USER
    else:
        return USER

#%%
def batch2TrainData(myVoc, sentences, rating, isSort = False, normalizeRating = False, testing=False):
    sentences_rating_pair = list()
    for index in range(len(sentences)):
        sentences_rating_pair.append(
            [
                sentences[index], rating[index], 
            ]
        )
    
    sentences_batch, rating_batch = [], []
    for pair in sentences_rating_pair:
        sentences_batch.append(pair[0])
        rating_batch.append(pair[1])

    inp, lengths = inputVar(
        sentences_batch,
        myVoc,
        testing)
    
    label = torch.tensor([val for val in rating_batch])
    # Wheather normalize rating
    if(normalizeRating):
        label = (label-1)/(5-1)

    return inp, lengths, label

#%%
def batch2LabelData(rating, this_asin, this_reviewerID, itemObj, userObj, normalizeRating = False):
    sentences_rating_pair = list()
    for index in range(len(rating)):
        sentences_rating_pair.append(
            [
                rating[index],
                this_asin[index], this_reviewerID[index]
            ]
        )

    rating_batch, this_asin_batch, this_reviewerID_batch  = [], [], []
    for pair in sentences_rating_pair:
        rating_batch.append(pair[0])
        this_asin_batch.append(itemObj.asin2index[pair[1]])
        this_reviewerID_batch.append(userObj.reviewerID2index[pair[2]])

    
    label = torch.tensor([val for val in rating_batch])
    # Wheather normalize rating
    if(normalizeRating):
        label = (label-1)/(5-1)

    # asin and reviewerID batch association
    asin_batch = torch.tensor([val for val in this_asin_batch])
    reviewerID_batch = torch.tensor([val for val in this_reviewerID_batch])

    return label, asin_batch, reviewerID_batch

#%%
"""
Create sentences & length encoding
"""
def GenerateTrainingBatches(USER, voc, num_of_reviews = 5 , batch_size = 5, testing=False):
    new_training_batches_sentences = list()
    new_training_batches_ratings = list()
    new_training_batches = list()
    num_of_batch_group = 0
    

    # for each user numberth reivews
    for review_ctr in range(0, num_of_reviews, 1):
        new_training_batch_sen = dict() # 40 (0~39)
        new_training_batch_rating = dict()
        training_batches = dict()
        for user_ctr in range(len(USER)):
            
            # Insert group encodeing
            if((user_ctr % batch_size == 0) and user_ctr>0):
                num_of_batch_group+=1
                
                # encode pre group
                training_batch = batch2TrainData(
                                    voc, 
                                    new_training_batch_sen[num_of_batch_group-1], 
                                    new_training_batch_rating[num_of_batch_group-1],
                                    isSort = False,
                                    normalizeRating = False,
                                    testing=testing
                                    )
                # training_batches[num_of_batch_group-1].append(training_batch)
                training_batches[num_of_batch_group-1] = training_batch


            this_user_sentence = USER[user_ctr].sentences[review_ctr]
            this_user_rating = USER[user_ctr].rating[review_ctr]

            if(num_of_batch_group not in new_training_batch_sen):
                new_training_batch_sen[num_of_batch_group] = []
                new_training_batch_rating[num_of_batch_group] = []
                # training_batches[num_of_batch_group] = []

            new_training_batch_sen[num_of_batch_group].append(this_user_sentence)
            new_training_batch_rating[num_of_batch_group].append(this_user_rating)    

            # Insert group encodeing (For Last group)
            if(user_ctr == (len(USER)-1)):
                num_of_batch_group+=1
                # encode pre group
                training_batch = batch2TrainData(
                                    voc, 
                                    new_training_batch_sen[num_of_batch_group-1], 
                                    new_training_batch_rating[num_of_batch_group-1],
                                    isSort = False,
                                    normalizeRating = False,
                                    testing=testing                                    
                                    )
                # training_batches[num_of_batch_group-1].append(training_batch)
                training_batches[num_of_batch_group-1] = training_batch            


        new_training_batches_sentences.append(new_training_batch_sen)
        new_training_batches_ratings.append(new_training_batch_rating)
        new_training_batches.append(training_batches)

        num_of_batch_group = 0
    return new_training_batches

#%%
def GenerateLabelEncoding(USER, num_of_reviews, num_of_rating):
    # num_of_rating = 1
    training_labels = dict()
    training_asins = dict()
    training_reviewerIDs = dict()

    for user_ctr in range(len(USER)):
        new_training_label = list()
        new_training_asin = list()
        new_training_reviewerID = list()

        # Create label (rating, asin, reviewers) structure
        for rating_ctr in range(num_of_reviews, num_of_reviews+num_of_rating, 1):
            
            this_traning_label = USER[user_ctr].rating[rating_ctr]
            new_training_label.append(this_traning_label)

            this_asin = USER[user_ctr].this_asin[rating_ctr]
            new_training_asin.append(this_asin)

            this_reviewerID = USER[user_ctr].this_reviewerID[rating_ctr]
            new_training_reviewerID.append(this_reviewerID)

        new_training_label, asin_batch, reviewerID_batch = batch2LabelData(new_training_label, new_training_asin, new_training_reviewerID, itemObj, userObj)
        
        training_labels[user_ctr] = new_training_label   
        training_asins[user_ctr] = asin_batch   
        training_reviewerIDs[user_ctr] = reviewerID_batch  

    return training_labels, training_asins, training_reviewerIDs
    
#%%
def GenerateBatchLabelCandidate(labels_, asins_, reviewerIDs_, batch_size):
    num_of_batch_group = 0
    batch_labels = dict()
    candidate_asins = dict()
    candidate_reviewerIDs = dict()

    batch_labels[num_of_batch_group] = list()
    candidate_asins[num_of_batch_group] = list()
    candidate_reviewerIDs[num_of_batch_group] = list()

    for idx in range(len(labels_)):
        if((idx % batch_size == 0) and idx > 0):
            num_of_batch_group+=1
            batch_labels[num_of_batch_group] = list()
            candidate_asins[num_of_batch_group] = list()
            candidate_reviewerIDs[num_of_batch_group] = list()
        
        batch_labels[num_of_batch_group].append(labels_[idx])
        candidate_asins[num_of_batch_group].append(asins_[idx])
        candidate_reviewerIDs[num_of_batch_group].append(reviewerIDs_[idx])

    return batch_labels, candidate_asins, candidate_reviewerIDs

#%%
def trainIteration(IntraGRU, InterGRU, IntraGRU_optimizer, InterGRU_optimizer, training_batches, 
    candidate_asins, candidate_reviewerIDs, training_batch_labels):

    group_loss=0

    for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): # how many batches
        for idx in range(len(training_batch_labels)):
            loss = 0
            InterGRU_optimizer.zero_grad()
            # Forward pass through HANN
            for reviews_ctr in range(len(training_batches)): # loop review 1 to 10

                IntraGRU_optimizer[reviews_ctr].zero_grad()

                current_batch = training_batches[reviews_ctr][batch_ctr]            
                input_variable, lengths, ratings = current_batch
                input_variable = input_variable.to(device)
                lengths = lengths.to(device)

                current_asins = torch.tensor(candidate_asins[idx][batch_ctr]).to(device)
                current_reviewerIDs = torch.tensor(candidate_reviewerIDs[idx][batch_ctr]).to(device)

                outputs, intra_hidden, intra_attn_score = IntraGRU[reviews_ctr](input_variable, lengths, 
                    current_asins, current_reviewerIDs)
                
                outputs = outputs.unsqueeze(0)

                if(reviews_ctr == 0):
                    interInput = outputs
                else:
                    interInput = torch.cat((interInput, outputs) , 0) 
            
            outputs, intra_hidden, inter_attn_score  = InterGRU(interInput, current_asins, current_reviewerIDs)
            outputs = outputs.squeeze(1)
            
            # Caculate loss 
            current_labels = torch.tensor(training_batch_labels[idx][batch_ctr]).to(device)

            err = (outputs*(5-1)+1) - current_labels
            loss = torch.mul(err, err)
            loss = loss.sum()/len(training_batches)
            
            # Perform backpropatation
            loss.backward()

            # Adjust model weights
            for reviews_ctr in range(len(training_batches)):
                IntraGRU_optimizer[reviews_ctr].step()
            InterGRU_optimizer.step()

            group_loss += loss

    return group_loss

#%%
def Train(myVoc, training_batches, candidate_asins, candidate_reviewerIDs, training_batch_labels, 
    directory, TrainEpoch=100, isStoreModel=False, WriteTrainLoss=False, store_every = 2):

    """
    Model setup
    """

    hidden_size = 300
    # Get asin and reviewerID from file
    asin, reviewerID = Read_Asin_Reviewer()
    # Initialize textual embeddings
    embedding = nn.Embedding(myVoc.num_words, hidden_size)
    # Initialize asin/reviewer embeddings
    asin_embedding = nn.Embedding(len(asin), hidden_size)
    reviewerID_embedding = nn.Embedding(len(reviewerID), hidden_size)    

    # Configure training/optimization
    learning_rate = 0.000001
    
    # Initialize IntraGRU models
    IntraGRU = list()
    # Initialize IntraGRU optimizers
    IntraGRU_optimizer = list()

    # Append GRU model asc
    for idx in range(num_of_reviews):    
        IntraGRU.append(IntraReviewGRU(hidden_size, embedding, asin_embedding, reviewerID_embedding))
        # Use appropriate device
        IntraGRU[idx] = IntraGRU[idx].to(device)
        IntraGRU[idx].train()

        # Initialize optimizers
        IntraGRU_optimizer.append(optim.Adam(IntraGRU[idx].parameters(), 
                lr=learning_rate, weight_decay=0.001)
            )

    # Initialize InterGRU models
    InterGRU = HANN(hidden_size, embedding, asin_embedding, reviewerID_embedding,
            n_layers=1, dropout=0, latentK = 64)
    # Use appropriate device
    InterGRU = InterGRU.to(device)
    InterGRU.train()
    # Initialize IntraGRU optimizers    
    InterGRU_optimizer = optim.Adam(InterGRU.parameters(), 
            lr=learning_rate, weight_decay=0.001)

    print('Models built and ready to go!')

    for Epoch in range(TrainEpoch):
        """
        Run multiple label for training HERE !!!! 
        """
        # Run a training iteration with batch
        group_loss = trainIteration(IntraGRU, InterGRU, IntraGRU_optimizer, InterGRU_optimizer, training_batches, 
            candidate_asins, candidate_reviewerIDs, training_batch_labels)

        """
        Run multiple label for training HERE !!!! 
        """
        num_of_iter = len(training_batches[0])*len(training_batch_labels)
        current_loss_average = group_loss/num_of_iter
        print('Epoch:{}\tSE:{}\t'.format(Epoch, current_loss_average))

        if(Epoch % store_every == 0 and isStoreModel):
            torch.save(InterGRU, R'ReviewsPrediction_Model/model/{}/InterGRU_epoch{}'.format(directory, Epoch))
            for idx__, IntraGRU__ in enumerate(IntraGRU):
                torch.save(IntraGRU__, R'ReviewsPrediction_Model/model/{}/IntraGRU_idx{}_epoch{}'.format(directory, idx__, Epoch))
            
        if WriteTrainLoss:
            with open(R'ReviewsPrediction_Model/Loss/{}/TrainingLoss.txt'.format(directory),'a') as file:
                file.write('Epoch:{}\tSE:{}\n'.format(Epoch, current_loss_average))        

#%%
def evaluate(IntraGRU, InterGRU, training_batches, validate_batch_labels, validate_asins, validate_reviewerIDs):
    group_loss = 0
    for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): #how many batches
        for idx in range(len(validate_batch_labels)):
            for reviews_ctr in range(len(training_batches)): #loop review 1 to 5
                
                current_batch = training_batches[reviews_ctr][batch_ctr]
                
                input_variable, lengths, ratings = current_batch
                input_variable = input_variable.to(device)
                lengths = lengths.to(device)

                current_asins = torch.tensor(validate_asins[idx][batch_ctr]).to(device)
                current_reviewerIDs = torch.tensor(validate_reviewerIDs[idx][batch_ctr]).to(device)

                with torch.no_grad():
                    outputs, intra_hidden, intra_attn_score = IntraGRU[reviews_ctr](input_variable, lengths, 
                        current_asins, current_reviewerIDs)
                    outputs = outputs.unsqueeze(0)

                    if(reviews_ctr == 0):
                        interInput = outputs
                    else:
                        interInput = torch.cat((interInput, outputs) , 0) 
            
            stop = 1
            with torch.no_grad():
                outputs, intra_hidden, inter_attn_score  = InterGRU(interInput, current_asins, current_reviewerIDs)
                outputs = outputs.squeeze(1)
            
            current_labels = torch.tensor(validate_batch_labels[idx][batch_ctr]).to(device)

            err = (outputs*(5-1)+1) - current_labels
            loss = torch.mul(err, err)
            # loss = torch.sqrt(loss)

            loss = loss.sum()/len(training_batches) # this batch avg. loss
            
            group_loss += loss

    num_of_iter = len(training_batches[0])*len(validate_batch_labels)
    RMSE = torch.sqrt(group_loss/num_of_iter)
    return RMSE

#%%
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
    
#%%
# res, itemObj, userObj = loadData(havingCount=20, LIMIT=2000)  # for elec.
res, itemObj, userObj = loadData(havingCount=15, LIMIT=1000, testing=False, table='toys_')  # for toys

#%%
voc, USER = Generate_Voc_User(res, batch_size=10, validation_batch_size=5)    # 
num_of_reviews = 9
batch_size = 4
num_of_rating = 3
num_of_validate = 3

#%%
for idx in range(len(USER)):
    if len(USER[idx].rating) <15:
        print('{} {}'.format(idx, len(USER[idx].rating)))    

#%% Generate training labels
training_batch_labels = list()
candidate_asins = list()
candidate_reviewerIDs = list()

for idx in range(0, num_of_rating, 1):

    training_labels, training_asins, training_reviewerIDs = GenerateLabelEncoding(USER, 
        num_of_reviews+idx, 1)
    
    _batch_labels, _asins, _reviewerIDs = GenerateBatchLabelCandidate(training_labels, training_asins, training_reviewerIDs, batch_size)
    
    training_batch_labels.append(_batch_labels)
    candidate_asins.append(_asins)
    candidate_reviewerIDs.append(_reviewerIDs)
#%% Generate validation labels
validate_batch_labels = list()
validate_asins = list()
validate_reviewerIDs = list()

for idx in range(0, num_of_validate, 1):

    validation_labels, validation_asins, validation_reviewerIDs = GenerateLabelEncoding(USER, 
        (num_of_reviews+num_of_rating)+idx , 1)
    
    _batch_labels, _asins, _reviewerIDs = GenerateBatchLabelCandidate(validation_labels, validation_asins, validation_reviewerIDs, batch_size)
    
    validate_batch_labels.append(_batch_labels)
    validate_asins.append(_asins)
    validate_reviewerIDs.append(_reviewerIDs)
#%%
training_batches = GenerateTrainingBatches(USER, voc, num_of_reviews=num_of_reviews, batch_size=batch_size)
#%% Training
directory = '1113_toys_lr1e06'
if(not True):
    Train(voc, training_batches, candidate_asins, candidate_reviewerIDs, training_batch_labels, 
        directory, TrainEpoch=100, isStoreModel=True, WriteTrainLoss=True, store_every = 2)

#%% Evaluation
if(not True):
    for Epoch in range(0, 101, 2):
        # Loading IntraGRU
        IntraGRU = list()
        for idx in range(num_of_reviews):
            model = torch.load(R'ReviewsPrediction_Model/model/{}/IntraGRU_idx{}_epoch{}'.format(directory, idx, Epoch))
            IntraGRU.append(model)

        # Loading InterGRU
        InterGRU = torch.load(R'ReviewsPrediction_Model/model/{}/InterGRU_epoch{}'.format(directory, Epoch))

        # evaluating
        RMSE = evaluate(IntraGRU, InterGRU, training_batches, validate_batch_labels, validate_asins, validate_reviewerIDs)
        print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))

        with open(R'ReviewsPrediction_Model/Loss/{}/ValidationLoss.txt'.format(directory),'a') as file:
            file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))    

#%%
"""
Testing
"""
#%% Loading esting data from database
res, itemObj, userObj = loadData(havingCount=20, LIMIT=200, testing=True, table='toys_')

#%%
USER = Generate_Voc_User(res, batch_size=10, validation_batch_size=5, generateVoc=False)
num_of_reviews = 9
batch_size = 4
num_of_rating = 3
#%%
len(USER), len(res)

#%% Generate training labels
testing_batch_labels = list()
candidate_asins = list()
candidate_reviewerIDs = list()

for idx in range(0, num_of_rating, 1):

    testing_labels, testing_asins, testing_reviewerIDs = GenerateLabelEncoding(USER, 
        num_of_reviews+idx, 1)
    
    _batch_labels, _asins, _reviewerIDs = GenerateBatchLabelCandidate(testing_labels, testing_asins, testing_reviewerIDs, batch_size)
    
    testing_batch_labels.append(_batch_labels)
    candidate_asins.append(_asins)
    candidate_reviewerIDs.append(_reviewerIDs)

# %%
testing_batches = GenerateTrainingBatches(USER, voc, num_of_reviews=num_of_reviews, batch_size=batch_size, testing=True)

#%% Evaluation (testing data)
directory = '1113_toys_lr1e06'
for Epoch in range(0, 101, 2):
    # Loading IntraGRU
    IntraGRU = list()
    for idx in range(num_of_reviews):
        model = torch.load(R'ReviewsPrediction_Model/model/{}/IntraGRU_idx{}_epoch{}'.format(directory, idx, Epoch))
        IntraGRU.append(model)

    # Loading InterGRU
    InterGRU = torch.load(R'ReviewsPrediction_Model/model/{}/InterGRU_epoch{}'.format(directory, Epoch))

    # evaluating
    RMSE = evaluate(IntraGRU, InterGRU, testing_batches, testing_batch_labels, candidate_asins, candidate_reviewerIDs)
    print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))

    with open(R'ReviewsPrediction_Model/Loss/{}/TestingLoss.txt'.format(directory),'a') as file:
        file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))    


# %%
