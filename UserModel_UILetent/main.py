#%%
# from UserModel_UILetent.preprocessing import Preprocess
# from UserModel_UILetent.model import HANN
from preprocessing import Preprocess
from model import  HANN
import tqdm
#%%
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

#%%
def trainIteration(InterGRU, InterGRU_optimizer, training_batches, 
    candidate_asins, candidate_reviewerIDs, training_batch_labels):

    group_loss=0

    for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): # how many batches
        # Run multiple label for training !!
        for idx in range(len(training_batch_labels)):
            loss = 0
            InterGRU_optimizer.zero_grad()
            # Forward pass through HANN
            for reviews_ctr in range(len(training_batches)): # loop review 1 to 10
                current_asins = training_batches[reviews_ctr][batch_ctr]
                current_asins = torch.tensor([val for val in current_asins]).to(device)
                current_asins = current_asins.unsqueeze(1)

                if(reviews_ctr==0):
                    asins_cat = current_asins
                else:
                    asins_cat = torch.cat((asins_cat, current_asins) , 1) 
            
            asins_cat = torch.transpose(asins_cat, 0, 1)

            outputs, intra_hidden  = InterGRU(asins_cat)
            outputs = outputs.squeeze(1)
            
            # Caculate loss 
            current_labels = torch.tensor(training_batch_labels[idx][batch_ctr]).to(device)

            err = (outputs*(5-1)+1) - current_labels
            loss = torch.mul(err, err)
            loss = torch.mean(loss, dim=0)
            
            # Perform backpropatation
            loss.backward()

            # Adjust model weights
            InterGRU_optimizer.step()

            group_loss += loss

    return group_loss

#%%
def Train(myVoc, table, training_batches, candidate_asins, candidate_reviewerIDs, training_batch_labels, 
    directory, validate_batch_labels, validate_asins, validate_reviewerIDs, 
    TrainEpoch=100, isStoreModel=False, WriteTrainLoss=False, store_every = 2):

    hidden_size = 300
    # Get asin and reviewerID from file
    asin, reviewerID = pre_work.Read_Asin_Reviewer(table)
    # Initialize textual embeddings
    embedding = nn.Embedding(myVoc.num_words, hidden_size)
    # Initialize asin/reviewer embeddings
    asin_embedding = nn.Embedding(len(asin), hidden_size)
    reviewerID_embedding = nn.Embedding(len(reviewerID), hidden_size)    

    # Configure training/optimization
    learning_rate = 0.00001  # elec
    
    # Initialize InterGRU models
    InterGRU = HANN(hidden_size, embedding, asin_embedding, reviewerID_embedding,
            n_layers=1, dropout=0, latentK = 64)
    # Use appropriate device
    InterGRU = InterGRU.to(device)
    InterGRU.train()
    # Initialize InterGRU optimizers    
    InterGRU_optimizer = optim.Adam(InterGRU.parameters(), 
            lr=learning_rate, weight_decay=0.001)

    print('Models built and ready to go!')

    for Epoch in range(TrainEpoch):

        # Run a training iteration with batch
        group_loss = trainIteration(InterGRU, InterGRU_optimizer, training_batches, 
            candidate_asins, candidate_reviewerIDs, training_batch_labels)

        num_of_iter = len(training_batches[0])*len(training_batch_labels)
        current_loss_average = group_loss/num_of_iter
        print('Epoch:{}\tSE:{}\t'.format(Epoch, current_loss_average))

        RMSE = evaluate(InterGRU, training_batches, validate_batch_labels, validate_asins, validate_reviewerIDs)

        print('\tMSE:{}\t'.format(RMSE))
        with open(R'ReviewsPrediction_Model/UserItemFeature/Loss/{}/ValidationLoss.txt'.format(directory),'a') as file:
            file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))        

        if(Epoch % store_every == 0 and isStoreModel):
            torch.save(InterGRU, R'ReviewsPrediction_Model/UserItemFeature/model/{}/InterGRU_epoch{}'.format(directory, Epoch))            

        if WriteTrainLoss:
            with open(R'ReviewsPrediction_Model/UserItemFeature/Loss/{}/TrainingLoss.txt'.format(directory),'a') as file:
                file.write('Epoch:{}\tSE:{}\n'.format(Epoch, current_loss_average))        

#%%
def evaluate(InterGRU, training_batches, validate_batch_labels, validate_asins, validate_reviewerIDs):
    group_loss = 0
    for batch_ctr in range(len(training_batches[0])): #how many batches
        for idx in range(len(validate_batch_labels)):
            for reviews_ctr in range(len(training_batches)): #loop review 1 to 5
                
                current_asins = training_batches[reviews_ctr][batch_ctr]
                current_asins = torch.tensor([val for val in current_asins]).to(device)
                current_asins = current_asins.unsqueeze(1)

                if(reviews_ctr==0):
                    asins_cat = current_asins
                else:
                    asins_cat = torch.cat((asins_cat, current_asins) , 1) 
            
            asins_cat = torch.transpose(asins_cat, 0, 1)

            with torch.no_grad():
                outputs, intra_hidden  = InterGRU(asins_cat)
                outputs = outputs.squeeze(1)

            
            current_labels = torch.tensor(validate_batch_labels[idx][batch_ctr]).to(device)

            err = (outputs*(5-1)+1) - current_labels
            loss = torch.mul(err, err)
            loss = torch.mean(loss, dim=0)            
            group_loss += loss

    num_of_iter = len(training_batches[0])*len(validate_batch_labels)
    RMSE = torch.sqrt(group_loss/num_of_iter)
    return RMSE

#%%
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
directory = '1201_clothing_bs16_lr1e05'

trainingEpoch = 50

trainOption = True
validationOption = True
testOption = True

# %%
"""
Setup for preprocessing
"""
pre_work = Preprocess()
num_of_reviews = 9
batch_size = 16
num_of_rating = 3
num_of_validate = 3

# %%
selectTable = 'clothing_'
res, itemObj, userObj = pre_work.loadData(havingCount=15, LIMIT=1000, testing=False, table='clothing_')  # for clothing.
# res, itemObj, userObj = loadData(havingCount=20, LIMIT=2000, testing=False, table='elec_')  # for elec.
# res, itemObj, userObj = pre_work.loadData(havingCount=15, LIMIT=1000, testing=False, table='toys_')  # for toys

voc, USER = pre_work.Generate_Voc_User(res, batch_size=10, validation_batch_size=5)

#%% Generate training labels
training_batch_labels = list()
candidate_asins = list()
candidate_reviewerIDs = list()

for idx in range(0, num_of_rating, 1):

    training_labels, training_asins, training_reviewerIDs = pre_work.GenerateLabelEncoding(USER, 
        num_of_reviews+idx, 1, itemObj, userObj)
    
    _batch_labels, _asins, _reviewerIDs = pre_work.GenerateBatchLabelCandidate(training_labels, training_asins, training_reviewerIDs, batch_size)
    
    training_batch_labels.append(_batch_labels)
    candidate_asins.append(_asins)
    candidate_reviewerIDs.append(_reviewerIDs)


#%% Generate validation labels
if(validationOption):
    validate_batch_labels = list()
    validate_asins = list()
    validate_reviewerIDs = list()

    for idx in range(0, num_of_validate, 1):

        validation_labels, validation_asins, validation_reviewerIDs = pre_work.GenerateLabelEncoding(USER, 
            (num_of_reviews+num_of_rating)+idx , 1, itemObj, userObj)
        
        _batch_labels, _asins, _reviewerIDs = pre_work.GenerateBatchLabelCandidate(validation_labels, validation_asins, validation_reviewerIDs, batch_size)
        
        validate_batch_labels.append(_batch_labels)
        validate_asins.append(_asins)
        validate_reviewerIDs.append(_reviewerIDs)

#%% Generate training batches
if(trainOption or validationOption):
    # Get training asin batches
    training_batches = pre_work.GenerateTrainingBatches(USER, itemObj, num_of_reviews=num_of_reviews, batch_size=batch_size)

# %%
if(trainOption):
    Train(voc, selectTable, training_batches, candidate_asins, candidate_reviewerIDs, training_batch_labels, 
        directory, validate_batch_labels, validate_asins, validate_reviewerIDs, 
        TrainEpoch=trainingEpoch, isStoreModel=True, WriteTrainLoss=True, store_every = 2)

#%% Evaluation
# if(validationOption):
#     for Epoch in range(0, trainingEpoch, 2):

#         # Loading InterGRU
#         InterGRU = torch.load(R'ReviewsPrediction_Model/UserItemFeature/model/{}/InterGRU_epoch{}'.format(directory, Epoch))

#         # evaluating
#         RMSE = evaluate(InterGRU, training_batches, validate_batch_labels, validate_asins, validate_reviewerIDs)
#         print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))

#         with open(R'ReviewsPrediction_Model/UserItemFeature/Loss/{}/ValidationLoss.txt'.format(directory),'a') as file:
#             file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))

#%% Testing
if(testOption):
    
    # Loading testing data from database
    # res, itemObj, userObj = pre_work.loadData(havingCount=20, LIMIT=500, testing=True, table='elec_')   # elec
    res, itemObj, userObj = pre_work.loadData(havingCount=15, LIMIT=200, testing=True, table='clothing_', withOutTable='clothing_training_1000')   # clothing
    # res, itemObj, userObj = pre_work.loadData(havingCount=15, LIMIT=400, testing=True, table='toys_', withOutTable='toys_training_1000')   # toys
    stop = 1
    USER = pre_work.Generate_Voc_User(res, batch_size=10, validation_batch_size=5, generateVoc=False)

    # Generate testing labels
    testing_batch_labels = list()
    candidate_asins = list()
    candidate_reviewerIDs = list()

    for idx in range(0, num_of_rating, 1):

        testing_labels, testing_asins, testing_reviewerIDs = pre_work.GenerateLabelEncoding(USER, 
            num_of_reviews+idx, 1, itemObj, userObj)
        
        _batch_labels, _asins, _reviewerIDs = pre_work.GenerateBatchLabelCandidate(testing_labels, testing_asins, testing_reviewerIDs, batch_size)
        
        testing_batch_labels.append(_batch_labels)
        candidate_asins.append(_asins)
        candidate_reviewerIDs.append(_reviewerIDs)

    # Generate testing batches
    testing_batches = pre_work.GenerateTrainingBatches(USER, itemObj, num_of_reviews=num_of_reviews, batch_size=batch_size)

    # Evaluation (testing data)
    for Epoch in range(0, trainingEpoch, 2):
        # Loading InterGRU
        InterGRU = torch.load(R'ReviewsPrediction_Model/UserItemFeature/model/{}/InterGRU_epoch{}'.format(directory, Epoch))

        # evaluating
        RMSE = evaluate(InterGRU, testing_batches, testing_batch_labels, candidate_asins, candidate_reviewerIDs)
        print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))

        with open(R'ReviewsPrediction_Model/UserItemFeature/Loss/{}/TestingLoss.txt'.format(directory),'a') as file:
            file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))    

#%%
InterGRU = torch.load(R'ReviewsPrediction_Model/UserItemFeature/model/{}/InterGRU_epoch{}'.format(directory, 48))

# %%
