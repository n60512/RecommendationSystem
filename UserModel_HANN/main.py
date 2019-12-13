#%%
# from UserModel_structure.preprocessing import Preprocess
# from UserModel_structure.model import IntraReviewGRU, HANN
import argparse
from preprocessing import Preprocess
from model import IntraReviewGRU, HANN
import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random

#%%
def randomSelectNoneReview(tensor_, itemEmbedding, type_='z'):
    # tensor_ size:[seq_size, batch_size, hidden_size]

    # Iterate each user(batch) to give 'Random Val.'
    for user in range(tensor_.size()[1]):

        # Random amount of reviews
        random_amount_of_reviews = random.randint(0, tensor_.size()[0]-1)

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
#%%
def trainIteration(IntraGRU, InterGRU, IntraGRU_optimizer, InterGRU_optimizer, training_batches, training_item_batches,
    candidate_items, candidate_users, training_batch_labels, isCatItemVec=False, RSNR=False):
    
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
                interInput = randomSelectNoneReview(interInput, InterGRU.itemEmbedding(interInput_asin), type_='v')

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

#%%
def Train(myVoc, table, training_batches, training_item_batches, candidate_items, candidate_users, training_batch_labels, 
    validate_batch_labels, validate_asins, validate_reviewerIDs , directory, TrainEpoch=100, latentK=32, intra_method ='dualFC', inter_method='dualFC',
     learning_rate = 0.00001, dropout=0, isStoreModel=False, WriteTrainLoss=False, store_every = 2, use_pretrain_item= False, isCatItemVec=False, RSNR=False):

    hidden_size = 300
    # Get asin and reviewerID from file
    asin, reviewerID = pre_work.Read_Asin_Reviewer(table)
    # Initialize textual embeddings
    embedding = nn.Embedding(myVoc.num_words, hidden_size)
    # Initialize asin/reviewer embeddings
    if(use_pretrain_item):
        asin_embedding = torch.load(R'PretrainingEmb/item_embedding_fromGRU.pth')
    else:
        asin_embedding = nn.Embedding(len(asin), hidden_size)
    reviewerID_embedding = nn.Embedding(len(reviewerID), hidden_size)    
    
    # Initialize IntraGRU models
    IntraGRU = list()
    # Initialize IntraGRU optimizers
    IntraGRU_optimizer = list()
    # Initialize IntraGRU optimizers groups
    intra_scheduler = list()

    # Append GRU model asc
    for idx in range(num_of_reviews):    
        IntraGRU.append(IntraReviewGRU(hidden_size, embedding, asin_embedding, reviewerID_embedding, method=intra_method))
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
            candidate_items, candidate_users, training_batch_labels, isCatItemVec=isCatItemVec, RSNR=RSNR)

        inter_scheduler.step()
        for idx in range(num_of_reviews):
            intra_scheduler[idx].step()

        num_of_iter = len(training_batches[0])*len(training_batch_labels)
        current_loss_average = group_loss/num_of_iter
        print('Epoch:{}\tSE:{}\t'.format(Epoch, current_loss_average))
        
        # evaluating
        RMSE = evaluate(IntraGRU, InterGRU, training_batches, training_asin_batches, 
            validate_batch_labels, validate_asins, validate_reviewerIDs, 
            isCatItemVec=isCatItemVec, RSNR=RSNR)

        print('\tMSE:{}\t'.format(RMSE))
        with open(R'ReviewsPrediction_Model/Loss/{}/ValidationLoss.txt'.format(directory),'a') as file:
            file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))


        if(Epoch % store_every == 0 and isStoreModel):
            torch.save(InterGRU, R'ReviewsPrediction_Model/model/{}/InterGRU_epoch{}'.format(directory, Epoch))
            for idx__, IntraGRU__ in enumerate(IntraGRU):
                torch.save(IntraGRU__, R'ReviewsPrediction_Model/model/{}/IntraGRU_idx{}_epoch{}'.format(directory, idx__, Epoch))
                    
        if WriteTrainLoss:
            with open(R'ReviewsPrediction_Model/Loss/{}/TrainingLoss.txt'.format(directory),'a') as file:
                file.write('Epoch:{}\tSE:{}\n'.format(Epoch, current_loss_average))        

#%%
def evaluate(IntraGRU, InterGRU, training_batches, training_asin_batches, validate_batch_labels, validate_asins, validate_reviewerIDs, isCatItemVec=False, RSNR=False):
    group_loss = 0
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
            
            if(RSNR):
                interInput = randomSelectNoneReview(interInput, InterGRU.itemEmbedding(interInput_asin), type_='v')

            with torch.no_grad():
                outputs, intra_hidden, inter_attn_score  = InterGRU(interInput, interInput_asin, current_asins, current_reviewerIDs)
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
if __name__ == "__main__":
    
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    directory = '1211_clothing_pre_cat_r4_bs40_lr1e05_lk32_dec20_dp0_nolimit_allGeneral'

    # %%
    """
    Setup for preprocessing
    """
    pre_work = Preprocess()

    # Parameters Setup
    trainOption = True
    validationOption = True
    testOption = True

    intra_method = 'general'
    inter_method = 'general'

    trainingEpoch = 50
    num_of_reviews = 4
    batch_size = 40
    num_of_rating = 3
    num_of_validate = 3
    store_every = 1
    latentK = 32
    learning_rate = 0.00001
    dropout = 0.5

    use_pretrain_item = True
    isCatItemVec = True
    Train_RSNR =not True
    Test_RSNR =not True
    

    selectTable = 'clothing_'
    res, itemObj, userObj = pre_work.loadData(havingCount=15, LIMIT=2000, testing=False, table='clothing_')  # for clothing.
    # res, itemObj, userObj = loadData(havingCount=20, LIMIT=2000, testing=False, table='elec_')  # for elec.
    # res, itemObj, userObj = pre_work.loadData(havingCount=15, LIMIT=1000, testing=False, table='toys_')  # for toys

    # Generate voc & User information
    voc, USER = pre_work.Generate_Voc_User(res, havingCount=15, limit_user=1000)

    # Generate training labels
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

    # Generate validation labels
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

    # Generate training batches
    if(trainOption):
        training_batches, training_asin_batches = pre_work.GenerateTrainingBatches(USER, itemObj, voc, num_of_reviews=num_of_reviews, batch_size=batch_size)

    if(trainOption):
        Train(voc, selectTable, training_batches, training_asin_batches, candidate_asins, candidate_reviewerIDs, training_batch_labels, 
            validate_batch_labels, validate_asins, validate_reviewerIDs, directory, TrainEpoch=trainingEpoch, latentK=latentK, intra_method=intra_method , inter_method=inter_method,
            learning_rate = learning_rate, dropout=dropout, isStoreModel=True, WriteTrainLoss=True, store_every = store_every, 
            use_pretrain_item=use_pretrain_item, isCatItemVec=isCatItemVec, RSNR=Train_RSNR)


    # Testing
    if(testOption):
        # Loading testing data from database
        # res, itemObj, userObj = pre_work.loadData(havingCount=20, LIMIT=500, testing=True, table='elec_')   # elec
        res, itemObj, userObj = pre_work.loadData(havingCount=15, LIMIT=200, testing=True, table='clothing_', withOutTable='clothing_training_1000')   # clothing
        # res, itemObj, userObj = pre_work.loadData(havingCount=15, LIMIT=400, testing=True, table='toys_', withOutTable='toys_training_1000')   # toys
        USER = pre_work.Generate_Voc_User(res, havingCount=15, generateVoc=False)

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
        testing_batches, testing_asin_batches = pre_work.GenerateTrainingBatches(USER, itemObj, voc, num_of_reviews=num_of_reviews, batch_size=batch_size, testing=True)

        # Evaluation (testing data)
        for Epoch in range(0, trainingEpoch, store_every):
            # Loading IntraGRU
            IntraGRU = list()
            for idx in range(num_of_reviews):
                model = torch.load(R'ReviewsPrediction_Model/model/{}/IntraGRU_idx{}_epoch{}'.format(directory, idx, Epoch))
                IntraGRU.append(model)

            # Loading InterGRU
            InterGRU = torch.load(R'ReviewsPrediction_Model/model/{}/InterGRU_epoch{}'.format(directory, Epoch))

            # evaluating
            RMSE = evaluate(IntraGRU, InterGRU, testing_batches, testing_asin_batches, testing_batch_labels, candidate_asins, candidate_reviewerIDs, 
                isCatItemVec=isCatItemVec, RSNR=Test_RSNR)
            print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))

            with open(R'ReviewsPrediction_Model/Loss/{}/TestingLoss.txt'.format(directory),'a') as file:
                file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))    

