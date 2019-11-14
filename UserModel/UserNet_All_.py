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
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0, latentK = 64):
        super(HANN, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.latentK = latentK

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding

        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_alpha = torch.nn.Linear(hidden_size, 1)

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.intra_review = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=dropout, 
                          bidirectional=True)

        self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear4 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_beta = torch.nn.Linear(hidden_size, 1)

        self.inter_review = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=dropout)
                          
        
        self.out128 = nn.Linear(hidden_size*2 , self.latentK*2)
        self.out64 = nn.Linear(self.latentK*2, self.latentK)
        self.out_ = nn.Linear(self.latentK, 1)

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

        intra_outputs = intra_outputs.unsqueeze(dim=1)

        # Forward pass through GRU
        outputs, current_hidden = self.inter_review(intra_outputs, hidden)
 
        # Sum bidirectional GRU outputs
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

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
        return sigmoid_outputs, current_hidden, intra_attn_score , inter_attn_score
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
def batch2TrainData(myVoc, sentences, rating, this_asin, this_reviewerID, itemObj, userObj, isSort = False, normalizeRating = False):
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
    # Wheather normalize rating
    if(normalizeRating):
        label = (label-1)/(5-1)

    # asin and reviewerID batch association
    asin_batch = torch.tensor([val for val in this_asin_batch])
    reviewerID_batch = torch.tensor([val for val in this_reviewerID_batch])

    return inp, lengths, label, asin_batch, reviewerID_batch

#%%
def Read_Asin_Reviewer():
    with open('asin_1.csv','r') as file:
        content = file.read()
        asin = content.split(',')
        print('asin count : {}'.format(len(asin)))

    with open('reviewerID.csv','r') as file:
        content = file.read()
        reviewerID = content.split(',')
        print('reviewerID count : {}'.format(len(reviewerID)))
    
    return asin, reviewerID

#%% Load dataset from database
def loadData(havingCount = 15, LIMIT=5000):

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
        '		HAVING COUNT(reviewerID) >= {} '.format(havingCount) +
        '		limit {} '.format(LIMIT) +
        '	) ' +
        'SELECT ' +
        'RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, ' +
        'review.`ID`, review.reviewerID , review.`asin`, review.overall, review.reviewText, review.unixReviewTime ' +
        'FROM review , metadata ' +
        'WHERE reviewerID IN (SELECT * FROM tenReviewsUp) ' +
        'AND review.`asin` = metadata.`asin` ' +
        'AND review.`reviewerID` != \'A28EMTHVF120XV\' ' +
        'AND review.`reviewerID` != \'A28HUBMSCXVQW0\' ' +
        'ORDER BY reviewerID,unixReviewTime ASC ;'
    )

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

#%% 根據使用者的回復數決定 batch
def GenerateBatches(res, itemObj, userObj, batch_size = 7, validation_batch_size=3):
    
    USER = list()
    LAST_USER = ''
    ctr = -1
    havingCount = batch_size + validation_batch_size

    # Creating Voc
    st = time.time()
    print('Creating Voc ...') 
    myVoc = Voc('Review')

    for index in tqdm.tqdm(range(len(res))):
        # Check is the next user or not
        if(LAST_USER != res[index]['reviewerID']):
            LAST_USER = res[index]['reviewerID']
            USER.append(PersonalHistory(res[index]['reviewerID']))
            ctr += 1   # add counter if change the user id
        
        if(res[index]['rank'] < havingCount + 1):
            current_sentence = normalizeString(res[index]['reviewText'])
            myVoc.addSentence(current_sentence) # myVoc add word !
            USER[ctr].addData(
                        current_sentence,
                        res[index]['overall'], 
                        res[index]['asin'],
                        res[index]['reviewerID']
                    )
            pass
        pass

    print('Voc creation complete. [{}]'.format(time.time()-st))
    

    st = time.time()
    print('Generating batches ...')     
    training_batches = dict()
    validation_batches = dict()

    for index in tqdm.tqdm(range(len(USER))):
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
        validation_batches[id_] = batch2TrainData(
                                    myVoc, 
                                    USER[index].sentences[batch_size:batch_size + validation_batch_size], 
                                    USER[index].rating[batch_size:batch_size + validation_batch_size], 
                                    USER[index].this_asin[batch_size:batch_size + validation_batch_size], 
                                    USER[index].this_reviewerID[batch_size:batch_size + validation_batch_size], 
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
            
    return training_batches, validation_batches, myVoc

#%% Train model
def train(training_batches, myVoc, directory, Train_Epoch=101, batch_size=15, WriteTrainLoss=True, label4training=3):
    # Configure models
    hidden_size = 300

    # Get asin and reviewerID from file
    asin, reviewerID = Read_Asin_Reviewer()

    # Initialize textual embeddings
    embedding = nn.Embedding(myVoc.num_words+1, hidden_size)

    # Initialize asin/reviewer embeddings
    asin_embedding = nn.Embedding(len(asin)+1, hidden_size)
    reviewerID_embedding = nn.Embedding(len(reviewerID)+1, hidden_size)

    # Initialize encoder & decoder models
    IntraGRU = HANN(hidden_size, embedding, asin_embedding, reviewerID_embedding, n_layers=1, dropout=0)
    # Use appropriate device
    IntraGRU = IntraGRU.to(device)
    print('Models built and ready to go!')

    # Configure training/optimization
    # learning_rate = 0.0000005
    # learning_rate = 0.000005
    learning_rate = 0.00001
    # learning_rate = 0.00002
    # learning_rate = 0.00004
    IntraGRU.train()

    # Initialize optimizers
    print('Building optimizers ...')
    IntraGRU_optimizer = optim.Adam(IntraGRU.parameters(), 
        lr=learning_rate, weight_decay=0.001)
    
    loss = 0
    current_loss = 0 
    store_every = 2

    # Assuming optimizer has two groups.
    # scheduler = optim.lr_scheduler.StepLR(IntraGRU_optimizer, 
    #     step_size=25, gamma=0.5)
    

    for Epoch in range(Train_Epoch):
        for key_userid, training_batch in tqdm.tqdm(training_batches.items()):

            input_variable, lengths, ratings , asin_index, reviewerID_index = training_batch

            # Set device options
            np_input_variable = input_variable.numpy()

            # Delete column at index -label4training to end
            np_input_variable = np.delete(np_input_variable,
                [val for val in range(batch_size, batch_size-label4training-1, -1)],
                axis=1)
            input_variable = torch.from_numpy(np_input_variable)
            input_variable = input_variable.to(device)

            lengths = lengths[:batch_size-label4training].to(device)

            # Last "label4training" values
            asin_index = asin_index[-label4training:].to(device)
            reviewerID_index = reviewerID_index[-label4training:].to(device)
            ratings = ratings[-label4training:].to(device)
                
            # User personal loss
            user_loss = 0
            user_row_count = 0

            for reviewerID, asin, true_rating in zip(asin_index, reviewerID_index, ratings):

                # Zero gradients
                IntraGRU_optimizer.zero_grad()

                reviewerID = reviewerID.unsqueeze(0) 
                asin = asin.unsqueeze(0)
                true_rating = true_rating.unsqueeze(0)

                # Forward pass through encoder
                outputs, intra_hidden, intra_attn_score, inter_attn_score = IntraGRU(input_variable, lengths, 
                    asin, reviewerID)
                    
                # Calculate Loss
                try:
                    err = (outputs*(5-1)+1) - true_rating
                    loss = torch.mul(err, err)
                    user_loss += loss
                        
                    loss.backward()
                    IntraGRU_optimizer.step()    
                    user_row_count+=1                
                    pass
                except RuntimeError as identifier:
                    with open(R'ReviewsPrediction_Model/model/{}/Error.txt'.format(directory),'a') as file:
                        file.write('User :{}\tErorr :{}\n'.format(key_userid, identifier))
                    pass

            user_loss = user_loss/user_row_count
            current_loss += user_loss
            pass
                
        current_loss_average = current_loss / len(training_batches)
        current_loss = 0

        # print('Epoch:{}\tSE:{}\tLR:{}'.format(Epoch, current_loss_average,scheduler.get_lr()))
        print('Epoch:{}\tSE:{}\t'.format(Epoch, current_loss_average))

        if Epoch % store_every == 0 and True:
            torch.save(IntraGRU, R'ReviewsPrediction_Model/model/{}/ReviewsPrediction_{}'.format(directory, Epoch))
        
        if WriteTrainLoss:
            with open(R'ReviewsPrediction_Model/Loss/{}/TrainingLoss.txt'.format(directory),'a') as file:
                file.write('Epoch:{}\tSE:{}\n'.format(Epoch, current_loss_average))
                # file.write('Epoch:{}\tSE:{}\tLR:{}\n'.format(Epoch, current_loss_average, scheduler.get_lr()))


# tensorboard --logdir=data/ --host localhost --port 8088

#%%
def js(index, attn_weight, directory):
    with open(R'{}/{}.html'.format(directory, index),'a') as file:
        text = "<!DOCTYPE html>\n<html>\n<body>\n<head>\n<script src='https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js'></script>\n</head><h1>Attn</h1><div id='text'>text goes here</div>\n"
        file.write(text)
        file.write('\n<script>\nvar words = [')

    for key, val in attn_weight.items():
        with open(R'{}/{}.html'.format(directory, index),'a') as file:
            file.write("{{\n'word': '{}',".format(key))
            file.write("'attention': {},\n}},".format(val))
    
    with open(R'{}/{}.html'.format(directory, index),'a') as file:
        file.write("];\n$('#text').html($.map(words, function(w) {\nreturn '<span style=\"background-color:hsl(360,100%,' + (w.attention * -50 + 100) + '%)\">' + w.word + ' </span>'\n}))\n</script>")


#%%
class UserAttnRecord():
    def __init__(self, userid):
        self.userid = userid
        self.intra_attn_score = list()
        self.inter_attn_score = list()
    
    def addIntraAttn(self, score):
        self.intra_attn_score.append(score)

    def addInterAttn(self, score):
        self.inter_attn_score.append(score)


#%%
class Evaluate(nn.Module):
    def __init__(self, model):
        super(Evaluate, self).__init__()
        self.model = model

    def forward(self, input_variable, lengths, asin_index, reviewerID_index):
        outputs, hidden , intra_attn_score, inter_attn_score = self.model(input_variable, lengths, 
            asin_index, reviewerID_index)
        return outputs, hidden , intra_attn_score, inter_attn_score

#%%
def Variables2Sentences(voc, input_variable, batch_size = 15):

    input_variable_np = input_variable.numpy()
    sentences = list()

    # Loop colunm 
    for colunm in range(batch_size):
        # Adding colunm sentence
        sentences.append(input_variable_np[:,colunm])

    return sentences

#%%
def evaluate(model , voc , itemObj, userObj, training_batches, validation_batches, batches_indexes = 7, output_count = 3, CalculateAttn = True):

    evaluator = Evaluate(model)
    current_loss = 0
    reviews_counter = 0

    # List for recording user attention values
    USER_ATTN_RECORD = list()

    for userid, validation_batch in tqdm.tqdm(validation_batches.items()):
        
        _, _, ratings , asin_index, reviewerID_index = validation_batch     # rating , asin for validation
        input_variable, lengths, _ , _, _ = training_batches[userid]        # user personal reviews record
        
        """
        Getting input_sentences
        """
        # Convert input_variable to input_sentences
        input_sentences = Variables2Sentences(voc, input_variable, batch_size= len(lengths))
        # remove zero value (PAD)
        input_sentences = [ input_sentences[index_][input_sentences[index_] != 0] 
                    for index_ in range(len(lengths)) ] 


        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        asin_index = asin_index.to(device)
        reviewerID_index = reviewerID_index.to(device)
        ratings = ratings.to(device)
        
        user_loss = 0
        for reviewerID, asin, true_rating in zip(asin_index, reviewerID_index, ratings):
            
            reviewerID = reviewerID.unsqueeze(0) 
            asin = asin.unsqueeze(0)
            true_rating = true_rating.unsqueeze(0)

            with torch.no_grad():
                outputs, hidden, intra_attn_score, inter_attn_score = evaluator(input_variable, lengths, 
                    asin, reviewerID)       
                
            outputs = outputs.squeeze(0)

            # Calculate loss (count from batches_indexes)    
            err = (outputs*(5-1)+1) - true_rating
            loss = torch.mul(err, err)  
            loss = math.sqrt(loss)
            user_loss += loss

            reviews_counter += 1
        
        current_loss += user_loss

        # Attention record
        if(CalculateAttn):

            # Construct user attention record object
            SingleUAR = UserAttnRecord(userid)

            # Adding inter attn score
            inter_attn_score_mean = torch.mean(inter_attn_score, dim = 0)
            SingleUAR.inter_attn_score = inter_attn_score_mean.squeeze().tolist()            

            # iterate input_sentences
            for index_ in range(len(input_sentences)):
                sentence_ = input_sentences[index_]
                sentence_attn_score = dict()
                word_count = 0
                
                # iterate each word of sentence_
                for word_number in sentence_:

                    # Convert index to word
                    word = voc.index2word[word_number]

                    # Adding word's attention score by intra_attn_score `colunm` data
                    # intra_attn_score[:,index_] : get number `index` colunm 
                    sentence_attn_score[word] = intra_attn_score[:,index_][word_count].item()
                    word_count += 1

                SingleUAR.addIntraAttn(sentence_attn_score)
                pass

            USER_ATTN_RECORD.append(SingleUAR)
            pass

    RMSE = current_loss/reviews_counter

    return RMSE, USER_ATTN_RECORD

#%%

# """
# Evalution
# """
# USE_CUDA = torch.cuda.is_available()
# device = torch.device("cuda" if USE_CUDA else "cpu")
    
# #%%
# res, itemObj, userObj = loadData(havingCount=10, LIMIT=200)  
# training_batches, validation_batches, myVoc = GenerateBatches(
#     res, 
#     itemObj, 
#     userObj,
#     batch_size = 7,
#     validation_batch_size=3
# )    
# #%%
# directory = R'test'
# train(training_batches, myVoc, directory, Train_Epoch=10, batch_size=7, WriteTrainLoss=True, label4training=2)

#%%



#%%
"""
Write attention result into html file and txt filr.
"""
def WriteAttention(USER_ATTN_RECORD):
    for userRecordObj in USER_ATTN_RECORD:
        # Create folder if not exists
        directory = ('AttentionVisualize/{}'.format(userRecordObj.userid))
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            print('folder : {} exist'.format(directory))
        
        index_ = 0
        for sentence in userRecordObj.intra_attn_score:
            js( index_, sentence, directory)   
            index_ += 1  
        
        with open(R'{}/Inter_Attn_Score.txt'.format(directory),'a') as file:
            count = 0
            for score in userRecordObj.inter_attn_score:
                file.write('Review {} : {}\n'.format(count, score))
                count += 1
        


#%%
if __name__ == "__main__":

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    directory = R'test'

    # Load in training batches
    res, itemObj, userObj = loadData(havingCount=25, LIMIT=200)  
    training_batches, validation_batches, myVoc = GenerateBatches(
        res, 
        itemObj, 
        userObj,
        batch_size = 20,
        validation_batch_size=5
    )    
    if(True):
        train(training_batches, myVoc, directory, Train_Epoch=101, batch_size=20, WriteTrainLoss=True, label4training=5)
        pass

    if(True):
        for Epoch in range(0,102,2):
            model = torch.load(R'ReviewsPrediction_Model/model/{}/ReviewsPrediction_{}'.format(directory, Epoch))
            RMSE, USER_ATTN_RECORD = evaluate(model ,  myVoc, itemObj, userObj, training_batches, validation_batches, 
                batches_indexes = 12, output_count=3, CalculateAttn=False)
            
            print('Epoch: {}\tRMSE: {}'.format(Epoch, RMSE))
            
            with open(R'ReviewsPrediction_Model/Loss/{}/ValidationLoss.txt'.format(directory),'a') as file:
                file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))    
    pass

#%%
a = 'B00L26YDA4,B00L3YHF6O,B00LGQ6HL8'
a.split(',')
len(a.split(',')),a.split(',')

# %%
