#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import re
import os
import unicodedata
import codecs
import itertools

import math
import numpy as np
import time
from DBconnector import DBConnection


#%%
class RNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.output_size = 64
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.out = nn.Linear(hidden_size,self.output_size)
        self.out_ = nn.Linear(self.output_size,1)

        self.attn = DotAttn(hidden_size)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        print(embedded.shape)
        stop = 1
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, current_hidden = self.gru(packed, hidden)
 
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)


        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(outputs, outputs)

        ## attn_weight : [5,1,200] , outputs.transpose(0, 1) :[5,200,500]
        ## bmm(batch martrix multiply) -> [1,200]*[200,500] = [1,500]
        context = attn_weights.bmm(outputs.transpose(0, 1)) 

        test = outputs.transpose(0, 1).transpose(1, 2)

        unsAtt = attn_weights.squeeze(1)
        new_outputs = unsAtt * outputs.transpose(0, 2)

        new_outputs = new_outputs.transpose(0, 2)

        new_outputs_sum = torch.sum(new_outputs , dim = 0)
        sixtyfourD_outputs = self.out(new_outputs_sum)

        oneD_outputs = self.out_(sixtyfourD_outputs)
        sigmoid_outputs = F.sigmoid(oneD_outputs)
        
        
        # new_outputs_softmax = F.softmax(new_outputs, dim=0)

        # z = torch.tensor([[0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [0.4, 0.5, 0.6, 0.2, 0.1, 0.3]])
        # z_softmax = F.softmax(z, dim=1).unsqueeze(1)
 
        # dot_Score = outputs * outputs
        # Score_Sum = torch.sum(outputs * outputs, dim=2)
        # Score_Sum_t = Score_Sum.t()
        # Score_Sum_softmax = F.softmax(Score_Sum_t, dim=1).unsqueeze(1)  

        a = 1 
        # Return output and final hidden state
        return sigmoid_outputs, current_hidden ,attn_weights

# Luong attention layer
class DotAttn(nn.Module):
    def __init__(self, hidden_size):
        super(DotAttn, self).__init__()
        self.hidden_size = hidden_size
        self.method = 'dot'

    def dot_score(self, hidden, rnn_output):
        return torch.sum(hidden * rnn_output, dim=2)

    def forward(self, hidden, rnn_output):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'dot':
            attn_energies = self.dot_score(hidden, rnn_output)
        # Transpose max_length and batch_size dimensions    
        # Transpose [seq_len,batch_size] to [batch_size,seq_len]    ## tensor size [5,200] 則 softmax function 對 200 維度做 softmax
        attn_energies = attn_energies.t()
        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class FeatureEmbedding(nn.Module):
    def __init__(self, rnn):
        super(FeatureEmbedding, self).__init__()
    
    def forward(self):
        return 0


class Evaluate(nn.Module):
    def __init__(self, rnn):
        super(Evaluate, self).__init__()
        self.rnn = rnn

    def forward(self, input_seq, input_lengths):
        rnn_outputs , rnn_hidden, attn_weights = self.rnn(input_seq, input_lengths)
        return rnn_outputs, attn_weights


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

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

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)



## convert all letters to lowercase 

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

    # return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    return [voc.word2index[word] for word in sentence_segment] + [EOS_token]

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
def batch2TrainData(myVoc, sentences, rating):
    sentences_rating_pair = list()
    for index in range(len(sentences)):
        sentences_rating_pair.append(
            [sentences[index],rating[index]]
        )

    sentences_rating_pair.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    # sentences.sort(key=lambda x: len(x.split(" ")), reverse=True)

    sentences_batch, rating_batch = [], []
    for pair in sentences_rating_pair:
        sentences_batch.append(pair[0])
        rating_batch.append(pair[1])

    inp, lengths = inputVar(
        sentences_batch,
        myVoc)
    
    label = torch.tensor([val for val in rating_batch])

    return inp, lengths, label

def loadData():

    sql = ('SELECT review.`asin`, review.overall, review.reviewText ' +
    ', metadata.title ' +
    'FROM review ' +
    'LEFT JOIN metadata ' +
    'ON review.`asin` = metadata.`asin` ' +
    'LIMIT 500;')

    conn = DBConnection()
    res = conn.selection(sql)

    single = False
    if(single):
        textSentence = res[0]['reviewText']

        print(textSentence + '\n')

        ## After normalize
        normalizeSentence = normalizeString(textSentence)
        print(normalizeSentence + '\n')

        # print(len(text))

        myVoc = Voc('Review')
        myVoc.addSentence(normalizeSentence)

        print(myVoc.word2index)
        print(myVoc.word2count)
        print(myVoc.index2word)
        print('\n ===================================')

    myVoc = Voc('Review')
    for row in res:
        current_sentence = row['reviewText']
        current_sentence = normalizeString(current_sentence)
        
        # text = current_sentence.split(' ')
        # print('Len:{}'.format(len(text)))
        # print(text[:20])
        
        myVoc.addSentence(current_sentence)
    
    # print(myVoc.word2index)
    # print(myVoc.word2count)
    # print(myVoc.index2word)

    sentences = list()
    rating = list()
    for index in range(len(res)):
        sentences.append(
            normalizeString(
            res[index]['reviewText']
            )
        )

        rating.append(
            res[index]['overall']
        )

    stop = 1

    training_batches = list()
    for index in range(0 , len(sentences)-1 , 5):
        sentence_rating_pair = sentences[index:index+5], rating[index:index+5]
        training_batches.append( batch2TrainData(myVoc, sentences[index:index+5], rating[index:index+5] ) )

    print(training_batches[0])

    stop=1
    return training_batches ,myVoc

def FeaturesEmbedded():
    
    pass

def train(training_batches, myVoc):
    # Configure models
    hidden_size = 500
    dropout = 0.1
    batch_size = 5


    # Initialize word embeddings
    embedding = nn.Embedding(myVoc.num_words, hidden_size)

    # Initialize encoder & decoder models
    rnn = RNN(hidden_size, embedding)
    # Use appropriate device
    rnn = rnn.to(device)
    print('Models built and ready to go!')

    # Configure training/optimization
    learning_rate = 0.000001
    rnn.train()

    # Initialize optimizers
    print('Building optimizers ...')
    rnn_optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

    # Zero gradients
    rnn_optimizer.zero_grad()

    criterion = nn.MSELoss()
    loss = 0

    for Epoch in range(100):

        for iteration in range(len(training_batches)):
            training_batch = training_batches[iteration]
            
            # input_variable = training_batch[0]
            # lengths = training_batch[1]
            input_variable, lengths, rating = training_batch

            # Set device options
            input_variable = input_variable.to(device)
            lengths = lengths.to(device)

            normalize_rating = (rating - 1)/ (5-1)
            normalize_rating = normalize_rating.to(device)

            stop = 1

            # Forward pass through encoder
            rnn_outputs, rnn_hidden, attn_weights = rnn(input_variable, lengths)

            loss = criterion(rnn_outputs, normalize_rating)
            loss.backward()

            rnn_optimizer.step()
        

        print('Epoch:{}\tLoss:{}'.format(Epoch,loss))

    torch.save(rnn, R'ReviewsPrediction_Model\ReviewsPrediction_01')

def evaluateData():

    sql = ('SELECT review.`asin`, review.overall, review.reviewText ' +
    ', metadata.title ' +
    'FROM review ' +
    'LEFT JOIN metadata ' +
    'ON review.`asin` = metadata.`asin` ' +
    'LIMIT 50;')

    conn = DBConnection()
    return conn.selection(sql)


def evaluate(rnn , voc , sentence):

    from VisualizeAttention import VisualizeAttn

    GraphGenerator = VisualizeAttn()

    evaluator = Evaluate(rnn)
    data = evaluateData()
    counter = 0
    for row in data:
        
        true_rating = row['overall']
        sentence = row['reviewText']

        # normalize String
        sentence = normalizeString(sentence)

        # words -> indexes
        indexes_batch = [indexesFromSentence(voc, sentence)]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)


        # print(input_batch, lengths)       
        output, attn_weights = evaluator(input_batch, lengths)
        predict_rating = (output*(5-1))+1

        text = list()
        print('========================================')
        print(true_rating, predict_rating.item())
        for val in indexes_batch[0]:
            text.append(voc.index2word[val] + ' ')
        
        attn_list = torch.squeeze(attn_weights).tolist()
        print(text)
        print(attn_list)
        print(len(text),len(attn_list))

        new_attn_list = list()
        for val in attn_list:
            new_attn_list.append(format(val * 100, '.5f'))

        print(new_attn_list)



        GraphGenerator.generate(text, new_attn_list, R"visualizeAttn\sample_{}.tex".format(counter), 'red')
        counter+=1

if __name__ == "__main__":

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    training_batches,myVoc = loadData()
    
    train(training_batches , myVoc)

    # rnn = torch.load(R'ReviewsPrediction_Model\ReviewsPrediction_01')
    # text = 'My son crewed my HD charger cord so I needed another one this is exactly like the one my son destroyed'
    # evaluate(rnn, myVoc , text)
    

    pass