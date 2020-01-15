
from utils import options, visulizeOutput
from utils.preprocessing import Preprocess
from utils.model import IntraReviewGRU, HANN
from visualization.attention_visualization import Visualization

import datetime

import tqdm
import torch
import torch.nn as nn
from torch import optim
import random

from gensim.models import KeyedVectors
import numpy as np

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
opt = options.GatherOptions().parse()

if(opt.user_pretrain_wordVec == 'Y'):
    filename = 'HNAE/data/clothing_festtext_subEmb.voc'
    pretrain_words = KeyedVectors.load_word2vec_format(filename, binary=False)

class UserAttnRecord():
    def __init__(self, userid):
        self.userid = userid
        self.intra_attn_score = list()
        self.inter_attn_score = list()
    
    def addIntraAttn(self, score):
        self.intra_attn_score.append(score)

    def addInterAttn(self, score):
        self.inter_attn_score.append(score)


def randomSelectNoneReview(tensor_, itemEmbedding, type_='z', randomSetup =-1):
    # tensor_ size:[seq_size, batch_size, hidden_size]

    # Iterate each user(batch) to give 'Random Val.'
    for user in range(tensor_.size()[1]):

        if(randomSetup==-1):
            # Random amount of reviews (max: all reviews)
            random_amount_of_reviews = random.randint(0, tensor_.size()[0]-1)
        else:
            random_amount_of_reviews = randomSetup

        for i_ in range(random_amount_of_reviews):
            select_random_seq = random.randint(0, tensor_.size()[0]-1)    
            # Give random hidden a NONE REVIEW
            if(type_=='z'):
                tensor_[select_random_seq][user] = torch.zeros(tensor_.size()[2], dtype=torch.float)
            elif(type_=='r'):
                pass
            elif(type_=='v'):
                tensor_[select_random_seq][user] = itemEmbedding[select_random_seq][user]
    return tensor_

def trainIteration(IntraGRU, InterGRU, IntraGRU_optimizer, InterGRU_optimizer, training_batches, training_item_batches,
    candidate_items, candidate_users, training_batch_labels, isCatItemVec=False, RSNR=False, randomSetup=-1):
    
    # Initialize this epoch loss
    epoch_loss = 0

    for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): # amount of batches
        # Run multiple label for training 
        for idx in range(len(training_batch_labels)):

            InterGRU_optimizer.zero_grad()

            # Forward pass through HANN
            for reviews_ctr in range(len(training_batches)): # iter. through reviews

                IntraGRU_optimizer[reviews_ctr].zero_grad()

                current_batch = training_batches[reviews_ctr][batch_ctr]            
                input_variable, lengths, ratings = current_batch
                input_variable = input_variable.to(device)
                lengths = lengths.to(device)

                current_asins = torch.tensor(candidate_items[idx][batch_ctr]).to(device)
                current_reviewerIDs = torch.tensor(candidate_users[idx][batch_ctr]).to(device)

                outputs, intra_hidden, intra_attn_score = IntraGRU[reviews_ctr](input_variable, lengths, 
                    current_asins, current_reviewerIDs)
                
                outputs = outputs.unsqueeze(0)

                if(isCatItemVec):
                    # Concat. asin feature
                    this_asins = training_item_batches[reviews_ctr][batch_ctr]
                    this_asins = torch.tensor([val for val in this_asins]).to(device)
                    this_asins = this_asins.unsqueeze(0)
                else:
                    interInput_asin = None

                if(reviews_ctr == 0):
                    interInput = outputs
                    if(isCatItemVec):
                        interInput_asin = this_asins
                else:
                    interInput = torch.cat((interInput, outputs) , 0) 
                    if(isCatItemVec):
                        interInput_asin = torch.cat((interInput_asin, this_asins) , 0) 
            
            if(RSNR):
                interInput = randomSelectNoneReview(interInput, InterGRU.itemEmbedding(interInput_asin), type_='v',randomSetup=randomSetup)

            outputs, intra_hidden, inter_attn_score  = InterGRU(interInput, interInput_asin, current_asins, current_reviewerIDs)
            outputs = outputs.squeeze(1)
            
            # Caculate loss 
            current_labels = torch.tensor(training_batch_labels[idx][batch_ctr]).to(device)

            err = (outputs*(5-1)+1) - current_labels
            loss = torch.mul(err, err)
            loss = torch.mean(loss, dim=0)
            
            # Perform backpropatation
            loss.backward()

            # Adjust model weights
            for reviews_ctr in range(len(training_batches)):
                IntraGRU_optimizer[reviews_ctr].step()
            InterGRU_optimizer.step()

            epoch_loss += loss

    return epoch_loss

def Train(myVoc, table, training_batches, training_item_batches, candidate_items, candidate_users, training_batch_labels, 
     directory, TrainEpoch=100, latentK=32, intra_method ='dualFC', inter_method='dualFC',
     learning_rate = 0.00001, dropout=0, isStoreModel=False, WriteTrainLoss=False, store_every = 2, use_pretrain_item= False, 
     isCatItemVec= True, RSNR=False, randomSetup=-1, pretrain_wordVec=None):

    hidden_size = 300
    # Get asin and reviewerID from file
    asin, reviewerID = pre_work.Read_Asin_Reviewer(table)

    # Initialize textual embeddings
    if(pretrain_wordVec != None):
        embedding = pretrain_wordVec
    else:
        embedding = nn.Embedding(myVoc.num_words, hidden_size)

    # Initialize asin/reviewer embeddings
    if(use_pretrain_item):
        asin_embedding = torch.load(R'PretrainingEmb/item_embedding_fromGRU.pth')
    else:
        asin_embedding = nn.Embedding(len(asin), hidden_size)
    reviewerID_embedding = nn.Embedding(len(reviewerID), hidden_size)    
    
    # Initialize IntraGRU models and optimizers
    IntraGRU = list()
    IntraGRU_optimizer = list()

    # Initialize IntraGRU optimizers groups
    intra_scheduler = list()

    # Append GRU model asc
    for idx in range(opt.num_of_reviews):    
        IntraGRU.append(IntraReviewGRU(hidden_size, embedding, asin_embedding, reviewerID_embedding,  
            latentK = latentK, method=intra_method))
        # Use appropriate device
        IntraGRU[idx] = IntraGRU[idx].to(device)
        IntraGRU[idx].train()

        # Initialize optimizers
        IntraGRU_optimizer.append(optim.Adam(IntraGRU[idx].parameters(), 
                lr=learning_rate, weight_decay=0.001)
            )
        
        # Assuming optimizer has two groups.
        intra_scheduler.append(optim.lr_scheduler.StepLR(IntraGRU_optimizer[idx], 
            step_size=20, gamma=0.3))

    
    # Initialize InterGRU models
    InterGRU = HANN(hidden_size, embedding, asin_embedding, reviewerID_embedding,
            n_layers=1, dropout=dropout, latentK = latentK, isCatItemVec=isCatItemVec , method=inter_method)

    # Use appropriate device
    InterGRU = InterGRU.to(device)
    InterGRU.train()
    # Initialize IntraGRU optimizers    
    InterGRU_optimizer = optim.Adam(InterGRU.parameters(), 
            lr=learning_rate, weight_decay=0.001)

    # Assuming optimizer has two groups.
    inter_scheduler = optim.lr_scheduler.StepLR(InterGRU_optimizer, 
        step_size=20, gamma=0.3)


    print('Models built and ready to go!')

    for Epoch in range(TrainEpoch):
        # Run a training iteration with batch
        group_loss = trainIteration(IntraGRU, InterGRU, IntraGRU_optimizer, InterGRU_optimizer, training_batches, training_item_batches, 
            candidate_items, candidate_users, training_batch_labels, isCatItemVec=isCatItemVec, RSNR=RSNR, randomSetup=randomSetup)

        inter_scheduler.step()
        for idx in range(opt.num_of_reviews):
            intra_scheduler[idx].step()

        num_of_iter = len(training_batches[0])*len(training_batch_labels)
        current_loss_average = group_loss/num_of_iter
        print('Epoch:{}\tSE:{}\t'.format(Epoch, current_loss_average))

        if(Epoch % store_every == 0 and isStoreModel):
            torch.save(InterGRU, R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))
            for idx__, IntraGRU__ in enumerate(IntraGRU):
                torch.save(IntraGRU__, R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx__, Epoch))
                    
        if WriteTrainLoss:
            with open(R'{}/Loss/TrainingLoss.txt'.format(opt.save_dir),'a') as file:
                file.write('Epoch:{}\tSE:{}\n'.format(Epoch, current_loss_average))        

def evaluate(IntraGRU, InterGRU, training_batches, training_asin_batches, validate_batch_labels, validate_asins, validate_reviewerIDs, 
    isCatItemVec=False, RSNR=False, randomSetup=-1, isWriteAttn=False, userObj=None):
    
    group_loss = 0
    # Voutput = visulizeOutput.WriteSentenceHeatmap(opt.save_dir, opt.num_of_reviews)
    AttnVisualize = Visualization(opt.save_dir, opt.num_of_reviews)

    # for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): #how many batches
    for batch_ctr in range(len(training_batches[0])): #how many batches
        for idx in range(len(validate_batch_labels)):
            for reviews_ctr in range(len(training_batches)): #loop review 1 to 5
                
                current_batch = training_batches[reviews_ctr][batch_ctr]
                
                input_variable, lengths, ratings = current_batch
                input_variable = input_variable.to(device)
                lengths = lengths.to(device)

                current_asins = torch.tensor(validate_asins[idx][batch_ctr]).to(device)
                current_reviewerIDs = torch.tensor(validate_reviewerIDs[idx][batch_ctr]).to(device)
        
                # Concat. asin feature
                this_asins = training_asin_batches[reviews_ctr][batch_ctr]
                this_asins = torch.tensor([val for val in this_asins]).to(device)
                this_asins = this_asins.unsqueeze(0)

                with torch.no_grad():
                    outputs, intra_hidden, intra_attn_score = IntraGRU[reviews_ctr](input_variable, lengths, 
                        current_asins, current_reviewerIDs)
                    outputs = outputs.unsqueeze(0)

                    if(reviews_ctr == 0):
                        interInput = outputs
                        interInput_asin = this_asins
                    else:
                        interInput = torch.cat((interInput, outputs) , 0) 
                        interInput_asin = torch.cat((interInput_asin, this_asins) , 0) 

                # Writing Intra-attention weight to .html file
                if(isWriteAttn):
                    for index_ , user_ in enumerate(current_reviewerIDs):

                        intra_attn_wts = intra_attn_score[:,index_].squeeze(1).tolist()
                        word_indexes = input_variable[:,index_].tolist()

                        # Voutput.js(intra_attn_wts, word_indexes, voc.index2word, reviews_ctr, fname='{}@{}'.format( userObj.index2reviewerID[user_.item()], reviews_ctr))
                        
                        sentence, weights = AttnVisualize.wdIndex2sentences(word_indexes, voc.index2word, intra_attn_wts)
                        AttnVisualize.createHTML(sentence, weights, reviews_ctr, 
                            fname='{}@{}'.format( userObj.index2reviewerID[user_.item()], reviews_ctr)
                            )
                            
            if(RSNR):
                interInput = randomSelectNoneReview(interInput, InterGRU.itemEmbedding(interInput_asin), type_='z',randomSetup=randomSetup)

            with torch.no_grad():
                outputs, intra_hidden, inter_attn_score  = InterGRU(interInput, interInput_asin, current_asins, current_reviewerIDs)
                outputs = outputs.squeeze(1)

            # Writing Inter-attention weight to .txt file
            if(isWriteAttn):
                for index_ , user_ in enumerate(current_reviewerIDs):
                    inter_attn_wts = inter_attn_score.squeeze(2)[:,index_].tolist()
                    with open('{}/VisualizeAttn/inter.txt'.format(opt.save_dir), 'a') as file:
                        file.write("=================================\nuser: {}\n".
                            format(userObj.index2reviewerID[user_.item()]))
                        for index__, val in enumerate(inter_attn_wts):
                            file.write('{} ,{}\n'.format(index__, val))           
            
            current_labels = torch.tensor(validate_batch_labels[idx][batch_ctr]).to(device)

            err = (outputs*(5-1)+1) - current_labels
            loss = torch.mul(err, err)
            loss = torch.mean(loss, dim=0)
            
            group_loss += loss

    num_of_iter = len(training_batches[0])*len(validate_batch_labels)
    RMSE = torch.sqrt(group_loss/num_of_iter)
    return RMSE


if __name__ == "__main__":


    pre_work = Preprocess(use_nltk_stopword=opt.use_nltk_stopword)
    print(opt.use_nltk_stopword)

    res, itemObj, userObj = pre_work.loadData(sqlfile=opt.sqlfile, testing=False, table= opt.selectTable)  # for clothing.

    # Generate voc & User information
    voc, USER = pre_work.Generate_Voc_User(res, having_interaction=opt.having_interactions)

    # Generate training labels
    training_batch_labels = list()
    candidate_asins = list()
    candidate_reviewerIDs = list()

    for idx in range(0, opt.num_of_rating, 1):
        stop = 1

        training_labels, training_asins, training_reviewerIDs = pre_work.GenerateLabelEncoding(USER, 
            opt.num_of_reviews+idx, 1, itemObj, userObj)
        
        _batch_labels, _asins, _reviewerIDs = pre_work.GenerateBatchLabelCandidate(training_labels, training_asins, training_reviewerIDs, opt.batchsize)
        
        training_batch_labels.append(_batch_labels)
        candidate_asins.append(_asins)
        candidate_reviewerIDs.append(_reviewerIDs)


    # pre-train words
    if(opt.user_pretrain_wordVec == 'Y'):
        weights_matrix = np.zeros((voc.num_words, 300))
        words_found = 0

        for index, word in voc.index2word.items():
            if(word == 'PAD'):
                weights_matrix[index] = np.zeros(300)   
            else:
                try: 
                    weights_matrix[index] = pretrain_words[word]
                    words_found += 1
                except KeyError as msg:
                    print(msg)
                    weights_matrix[index] = np.random.uniform(low=-1, high=1, size=(300))

        weight_tensor = torch.FloatTensor(weights_matrix)
        pretrain_wordVec = nn.Embedding.from_pretrained(weight_tensor).to(device)
    else:
        pretrain_wordVec = None


    # Generate training batches
    if(opt.mode == "train" or opt.mode == "both"):
        training_batches, training_asin_batches = pre_work.GenerateTrainingBatches(USER, itemObj, voc, num_of_reviews=opt.num_of_reviews, batch_size=opt.batchsize)

        Train(voc, opt.selectTable, training_batches, training_asin_batches, candidate_asins, candidate_reviewerIDs, training_batch_labels, 
            opt.save_dir, TrainEpoch=opt.epoch, latentK=opt.latentK, intra_method=opt.intra_attn_method , inter_method=opt.inter_attn_method,
            learning_rate = opt.lr, dropout=0, isStoreModel=True, WriteTrainLoss=True, store_every = opt.save_model_freq, 
            use_pretrain_item=False, isCatItemVec=True, RSNR=False, pretrain_wordVec=pretrain_wordVec)

    # Generate testing batches
    if(opt.mode == "test" or opt.mode == "showAttn" or opt.mode == "both"):

        # Loading testing data from database
        res, itemObj, userObj = pre_work.loadData(sqlfile=opt.sqlfile, testing=True, table=opt.selectTable)   # clothing
        USER = pre_work.Generate_Voc_User(res, having_interaction=opt.having_interactions, generateVoc=False)

        # Generate testing labels
        testing_batch_labels = list()
        candidate_asins = list()
        candidate_reviewerIDs = list()

        for idx in range(0, opt.num_of_rating, 1):

            testing_labels, testing_asins, testing_reviewerIDs = pre_work.GenerateLabelEncoding(USER, 
                opt.num_of_reviews+idx, 1, itemObj, userObj)
            
            _batch_labels, _asins, _reviewerIDs = pre_work.GenerateBatchLabelCandidate(testing_labels, testing_asins, testing_reviewerIDs, opt.batchsize)
            
            testing_batch_labels.append(_batch_labels)
            candidate_asins.append(_asins)
            candidate_reviewerIDs.append(_reviewerIDs)

        # Generate testing batches
        testing_batches, testing_asin_batches = pre_work.GenerateTrainingBatches(USER, itemObj, voc, 
            num_of_reviews=opt.num_of_reviews, batch_size=opt.batchsize, testing=True)


    # Testing
    if(opt.mode == "test" or opt.mode == "both"):

        # Evaluation (testing data)
        for Epoch in range(0, opt.epoch, opt.save_model_freq):
            # Loading IntraGRU
            IntraGRU = list()
            for idx in range(opt.num_of_reviews):
                model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, Epoch))
                IntraGRU.append(model)

            # Loading InterGRU
            InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))

            # evaluating
            RMSE = evaluate(IntraGRU, InterGRU, testing_batches, testing_asin_batches, testing_batch_labels, candidate_asins, candidate_reviewerIDs, 
                isCatItemVec=True, RSNR=False, randomSetup=-1)
            print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))

            with open(R'{}/Loss/TestingLoss.txt'.format(opt.save_dir),'a') as file:
                file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))    

    stop = 1

    # Testing (with showing attention weight)
    if(opt.mode == "showAttn"):
        # Loading IntraGRU
        IntraGRU = list()
        for idx in range(opt.num_of_reviews):
            model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, opt.visulize_attn_epoch))
            IntraGRU.append(model)

        # Loading InterGRU
        InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, opt.visulize_attn_epoch))

        # evaluating
        RMSE = evaluate(IntraGRU, InterGRU, testing_batches, testing_asin_batches, testing_batch_labels, candidate_asins, candidate_reviewerIDs, 
            isCatItemVec=True, RSNR=False, randomSetup=-1, isWriteAttn=True, userObj=userObj)
