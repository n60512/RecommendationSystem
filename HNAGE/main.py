from utils import options, visulizeOutput
from utils.preprocessing import Preprocess
from utils.model import IntraReviewGRU, HANN, DecoderGRU
from visualization.attention_visualization import Visualization

import datetime
import tqdm
import torch
import torch.nn as nn
from torch import optim
import random

from gensim.models import KeyedVectors
import numpy as np

# Use cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
opt = options.GatherOptions().parse()

# If use pre-train word vector , load .vec
if(opt.use_pretrain_wordVec == 'Y'):
    filename = 'HNAGE/data/{}festtext_subEmb.vec'.format(opt.selectTable)
    pretrain_words = KeyedVectors.load_word2vec_format(filename, binary=False)



# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class UserAttnRecord():
    def __init__(self, userid):
        self.userid = userid
        self.intra_attn_score = list()
        self.inter_attn_score = list()
    
    def addIntraAttn(self, score):
        self.intra_attn_score.append(score)

    def addInterAttn(self, score):
        self.inter_attn_score.append(score)

def trainIteration(IntraGRU, InterGRU, DecoderModel, DecoderModel_optimizer, 
    training_batches, training_item_batches, candidate_items, candidate_users, training_batch_labels, label_sen_batch,
    isCatItemVec=False, IntraGRU_optimizer=None, InterGRU_optimizer=None):
    
    # Initialize this epoch loss
    hann_epoch_loss = 0
    decoder_epoch_loss = 0

    for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): # amount of batches
        # Run multiple label for training 
        for idx in range(len(training_batch_labels)):
            
            # If turning HANN
            if(opt.tuning_HANN == 'Y'):
                InterGRU_optimizer.zero_grad()
                for reviews_ctr in range(len(training_batches)): # iter. through reviews
                    IntraGRU_optimizer[reviews_ctr].zero_grad()

            # Forward pass through HANN
            for reviews_ctr in range(len(training_batches)): # iter. through reviews
                
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
            
            outputs, inter_hidden, inter_attn_score  = InterGRU(interInput, interInput_asin, current_asins, current_reviewerIDs)
            outputs = outputs.squeeze(1)



            # Caculate Square loss of HANN 
            current_rating_labels = torch.tensor(training_batch_labels[idx][batch_ctr]).to(device)
            err = (outputs*(5-1)+1) - current_rating_labels

            HANN_loss = torch.mul(err, err)
            HANN_loss = torch.mean(HANN_loss, dim=0)

            # HANN loss of this epoch
            hann_epoch_loss += HANN_loss

            """
            Runing Decoder
            """
            # Ground true sentences
            target_batch = label_sen_batch[0][batch_ctr]
            target_variable, target_len, _ = target_batch 
            max_target_len = max(target_len)
            target_variable = target_variable.to(device)  

            # Create initial decoder input (start with SOS tokens for each sentence)
            decoder_input = torch.LongTensor([[SOS_token for _ in range(opt.batchsize)]])
            decoder_input = decoder_input.to(device)   


            # Set initial decoder hidden state to the inter_hidden's final hidden state
            criterion = nn.NLLLoss()
            decoder_loss = 0
            decoder_hidden = inter_hidden

            # Determine if we are using teacher forcing this iteration
            use_teacher_forcing = True if random.random() < opt.teacher_forcing_ratio else False

            # Forward batch of sequences through decoder one time step at a time
            if use_teacher_forcing:
                for t in range(max_target_len):
                    decoder_output, decoder_hidden = DecoderModel(
                        decoder_input, decoder_hidden
                    )
                    # Teacher forcing: next input is current target
                    decoder_input = target_variable[t].view(1, -1)  # get the row(word) of sentences

                    # Calculate and accumulate loss
                    nll_loss = criterion(decoder_output, target_variable[t])
                    decoder_loss += nll_loss
            else:
                for t in range(max_target_len):
                    decoder_output, decoder_hidden = DecoderModel(
                        decoder_input, decoder_hidden
                    )
                    # No teacher forcing: next input is decoder's own current output
                    _, topi = decoder_output.topk(1)

                    decoder_input = torch.LongTensor([[topi[i][0] for i in range(opt.batchsize)]])
                    decoder_input = decoder_input.to(device)

                    # Calculate and accumulate loss
                    nll_loss = criterion(decoder_output, target_variable[t])
                    decoder_loss += nll_loss
            

            loss = HANN_loss + decoder_loss

            # Perform backpropatation
            loss.backward()

            # Clip gradients: gradients are modified in place
            for reviews_ctr in range(len(training_batches)):            
                _ = nn.utils.clip_grad_norm_(IntraGRU[reviews_ctr].parameters(), opt.clip)
            _ = nn.utils.clip_grad_norm_(InterGRU.parameters(), opt.clip)


            # If turning HANN
            if(opt.tuning_HANN == 'Y'):
                # Adjust `HANN` model weights
                for reviews_ctr in range(len(training_batches)):
                    IntraGRU_optimizer[reviews_ctr].step()
                InterGRU_optimizer.step()

            # Adjust Decoder model weights
            DecoderModel_optimizer.step()

            # decoder loss of this epoch
            decoder_epoch_loss += decoder_loss.item()/float(max_target_len)

    stop = 1
    return hann_epoch_loss, decoder_epoch_loss

def Train(myVoc, table, training_batches, training_item_batches, candidate_items, candidate_users, training_batch_labels, label_sen_batch, 
     directory, TrainEpoch=100, latentK=32, hidden_size = 300, intra_method ='dualFC', inter_method='dualFC',
     learning_rate = 0.00001, dropout=0, isStoreModel=False, isStoreCheckPts=False, WriteTrainLoss=False, store_every = 2, use_pretrain_item= False, 
     isCatItemVec= True, pretrain_wordVec=None):

    
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
    

    # Initialize IntraGRU models
    IntraGRU = list()

    # IF USING PRETRAIN HANN
    if(opt.use_pretrain_HANN == 'Y'):
        inter_model_path = 'HNAGE/data/pretrain_hann/InterGRU_epoch24'

        for idx in range(opt.num_of_reviews):
            intra_model_path = 'HNAGE/data/pretrain_hann/IntraGRU_idx{}_epoch24'.format(idx)
            model = torch.load(intra_model_path)
            IntraGRU.append(model)

        # Loading InterGRU
        InterGRU = torch.load(inter_model_path)

        # Use appropriate device
        InterGRU = InterGRU.to(device)
        for idx in range(opt.num_of_reviews):    
            IntraGRU[idx] = IntraGRU[idx].to(device)
        
    else:
        # Append GRU model asc
        for idx in range(opt.num_of_reviews):    
            IntraGRU.append(IntraReviewGRU(hidden_size, embedding, asin_embedding, reviewerID_embedding,  
                latentK = latentK, method=intra_method))
            # Use appropriate device
            IntraGRU[idx] = IntraGRU[idx].to(device)
            IntraGRU[idx].train()
        
        # Initialize InterGRU models
        InterGRU = HANN(hidden_size, embedding, asin_embedding, reviewerID_embedding,
                n_layers=1, dropout=dropout, latentK = latentK, isCatItemVec=isCatItemVec , method=inter_method)

        # Use appropriate device
        InterGRU = InterGRU.to(device)
        InterGRU.train()


    # Initialize IntraGRU optimizers groups
    intra_scheduler = list()


    # IF tuning HANN
    if(opt.tuning_HANN == 'Y'):
        

        # Initialize `INTRA` model optimizers
        IntraGRU_optimizer = list()

        for idx in range(opt.num_of_reviews):    
            # Initialize optimizers
            IntraGRU_optimizer.append(optim.AdamW(IntraGRU[idx].parameters(), 
                    lr=learning_rate, weight_decay=0.001)
                )
            # Assuming optimizer has two groups.
            intra_scheduler.append(optim.lr_scheduler.StepLR(IntraGRU_optimizer[idx], 
                step_size=20, gamma=0.3))


        # Initialize `INTER` model optimizers    
        InterGRU_optimizer = optim.AdamW(InterGRU.parameters(), 
                lr=learning_rate, weight_decay=0.001)
        # Assuming optimizer has two groups.
        inter_scheduler = optim.lr_scheduler.StepLR(InterGRU_optimizer, 
            step_size=10, gamma=0.3)                
    
    else:
        IntraGRU_optimizer = None
        InterGRU_optimizer = None


    # Initialize DecoderGRU models and optimizers
    DecoderModel = DecoderGRU(embedding, hidden_size, myVoc.num_words, n_layers=1, dropout=dropout)
    # Use appropriate device
    DecoderModel = DecoderModel.to(device)
    DecoderModel.train()
    # Initialize DecoderGRU optimizers    
    DecoderModel_optimizer = optim.AdamW(DecoderModel.parameters(), 
            lr=learning_rate * opt.decoder_learning_ratio, 
            weight_decay=0.001)    

    print('Models built and ready to go!')

    for Epoch in range(TrainEpoch):
        # Run a training iteration with batch
        hann_group_loss, decoder_group_loss = trainIteration(IntraGRU, InterGRU, DecoderModel, DecoderModel_optimizer, 
            training_batches, training_item_batches, candidate_items, candidate_users, training_batch_labels, label_sen_batch, 
            isCatItemVec=isCatItemVec, IntraGRU_optimizer=IntraGRU_optimizer, InterGRU_optimizer=InterGRU_optimizer)

        # IF tuning HANN
        if(opt.tuning_HANN == 'Y'):
            inter_scheduler.step()
            for idx in range(opt.num_of_reviews):
                intra_scheduler[idx].step()

        num_of_iter = len(training_batches[0])*len(training_batch_labels)
        
        hann_loss_average = hann_group_loss/num_of_iter
        decoder_loss_average = decoder_group_loss/num_of_iter

        print('Epoch:{}\tHANN(SE):{}\tNNL:{}\t'.format(Epoch, hann_loss_average, decoder_loss_average))

        if(Epoch % store_every == 0 and isStoreModel):
            torch.save(InterGRU, R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))
            torch.save(DecoderModel, R'{}/Model/DecoderModel_epoch{}'.format(opt.save_dir, Epoch))
            for idx__, IntraGRU__ in enumerate(IntraGRU):
                torch.save(IntraGRU__, R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx__, Epoch))
                    
        if WriteTrainLoss:
            with open(R'{}/Loss/TrainingLoss.txt'.format(opt.save_dir),'a') as file:
                file.write('Epoch:{}\tHANN(SE):{}\tNNL:{}\n'.format(Epoch, hann_loss_average, decoder_loss_average))

        # Save checkpoint
        if (Epoch % store_every == 0 and isStoreCheckPts):
            # Store intra-GRU model
            for idx__, IntraGRU__ in enumerate(IntraGRU):
                state = {
                    'epoch': Epoch,
                    'num_of_review': idx__,
                    'intra{}'.format(idx__): IntraGRU__.state_dict(),
                    'intra{}_opt'.format(idx__): IntraGRU_optimizer[idx__].state_dict(),
                    'train_loss': current_loss_average,
                    'voc_dict': myVoc.__dict__,
                    'embedding': embedding.state_dict()
                }
                torch.save(state, R'{}/checkpts/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx__, Epoch))
            
            # Store inter-GRU model
            state = {
                'epoch': Epoch,
                'inter': InterGRU.state_dict(),
                'inter_opt': InterGRU_optimizer.state_dict(),
                'train_loss': current_loss_average,
                'voc_dict': myVoc.__dict__,
                'embedding': embedding.state_dict()
            }
            torch.save(state, R'{}/checkpts/InterGRU_epoch{}'.format(opt.save_dir, Epoch))


def evaluate(IntraGRU, InterGRU, DecoderModel, training_batches, training_asin_batches, validate_batch_labels, validate_asins, validate_reviewerIDs, testing_sen_batches,
    isCatItemVec=False, isWriteAttn=False, userObj=None, itemObj=None, voc=None):
    
    AttnVisualize = Visualization(opt.save_dir, opt.num_of_reviews)

    tokens_dict = dict()
    scores_dict = dict()

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
                        
                        sentence, weights = AttnVisualize.wdIndex2sentences(word_indexes, voc.index2word, intra_attn_wts)
                        AttnVisualize.createHTML(sentence, weights, reviews_ctr, 
                            fname='{}@{}'.format( userObj.index2reviewerID[user_.item()], reviews_ctr)
                            )
                            
            with torch.no_grad():
                outputs, inter_hidden, inter_attn_score  = InterGRU(interInput, interInput_asin, current_asins, current_reviewerIDs)
                outputs = outputs.squeeze(1)

            
            # Caculate Square loss of HANN 
            current_rating_labels = torch.tensor(validate_batch_labels[idx][batch_ctr]).to(device)
            predict_rating = (outputs*(5-1)+1)
            err = predict_rating - current_rating_labels

            # Writing Inter-attention weight to .txt file
            if(isWriteAttn):
                for index_ , user_ in enumerate(current_reviewerIDs):
                    inter_attn_wts = inter_attn_score.squeeze(2)[:,index_].tolist()
                    with open('{}/VisualizeAttn/inter.txt'.format(opt.save_dir), 'a') as file:
                        file.write("=================================\nuser: {}\n".
                            format(userObj.index2reviewerID[user_.item()]))
                        for index__, val in enumerate(inter_attn_wts):
                            file.write('{} ,{}\n'.format(index__, val))           
            

            """
            Greedy Search Strategy Decoder
            """

            # Create initial decoder input (start with SOS tokens for each sentence)
            decoder_input = torch.LongTensor([[SOS_token for _ in range(opt.batchsize)]])
            decoder_input = decoder_input.to(device)    

            # Set initial decoder hidden state to the inter_hidden's final hidden state
            criterion = nn.NLLLoss()
            loss = 0
            decoder_hidden = inter_hidden

            # Ground true sentences
            target_batch = testing_sen_batches[0][batch_ctr]
            target_variable, target_len, _ = target_batch   
            max_target_len = max(target_len)
            target_variable = target_variable.to(device)  


            # Initialize tensors to append decoded words to
            all_tokens = torch.zeros([0], device=device, dtype=torch.long)
            all_scores = torch.zeros([0], device=device)            

            
            for t in range(max_target_len):
                decoder_output, decoder_hidden = DecoderModel(
                    decoder_input, decoder_hidden
                )
                # No teacher forcing: next input is decoder's own current output
                decoder_scores_, topi = decoder_output.topk(1)

                decoder_input = torch.LongTensor([[topi[i][0] for i in range(opt.batchsize)]])
                decoder_input = decoder_input.to(device)

                ds, di = torch.max(decoder_output, dim=1)


                # Record token and score
                all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                all_scores = torch.cat((all_scores, torch.t(decoder_scores_)), dim=0)                

                # Calculate and accumulate loss
                nll_loss = criterion(decoder_output, target_variable[t])
                loss += nll_loss


            for index_ , user_ in enumerate(current_reviewerIDs):
                asin_ = current_asins[index_]

                current_user_tokens = all_tokens[:,index_].tolist()
                decoded_words = [voc.index2word[token] for token in current_user_tokens if token != 0]
                predict_rating, current_rating_labels[index_].item()


                generate_text = str.format('=========================\nUserid & asin:{},{}\nPredict:{:10.3f}\nRating:{:10.3f}\nGenerate: {}\n'.format(
                    userObj.index2reviewerID[user_.item()], 
                    itemObj.index2asin[asin_.item()],
                    predict_rating[index_].item(),
                    current_rating_labels[index_].item(),
                    ' '.join(decoded_words)))

                current_user_sen = target_variable[:,index_].tolist()
                origin_sen = [voc.index2word[token] for token in current_user_sen if token != 0]

                generate_text = generate_text + str.format('Origin: {}\n'.format(' '.join(origin_sen)))

                if opt.test_on_traindata == "Y":
                    fpath = R'{}/GenerateSentences/on_train/sentences_ep{}.txt'.format(opt.save_dir, opt.epoch)
                elif opt.test_on_traindata == "N":
                    fpath = R'{}/GenerateSentences/on_test/sentences_ep{}.txt'.format(opt.save_dir, opt.epoch)

                with open(fpath,'a') as file:
                    file.write(generate_text)     

    return tokens_dict, scores_dict


def evaluate_RMSE(IntraGRU, InterGRU, DecoderModel, training_batches, training_asin_batches, validate_batch_labels, validate_asins, validate_reviewerIDs, testing_sen_batches,
    isCatItemVec=False, isWriteAttn=False, userObj=None, itemObj=None, voc=None):
    
    AttnVisualize = Visualization(opt.save_dir, opt.num_of_reviews)
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

                # Writing Intra-attention weight to .html file
                if(isWriteAttn):
                    for index_ , user_ in enumerate(current_reviewerIDs):

                        intra_attn_wts = intra_attn_score[:,index_].squeeze(1).tolist()
                        word_indexes = input_variable[:,index_].tolist()
                        
                        sentence, weights = AttnVisualize.wdIndex2sentences(word_indexes, voc.index2word, intra_attn_wts)
                        AttnVisualize.createHTML(sentence, weights, reviews_ctr, 
                            fname='{}@{}'.format( userObj.index2reviewerID[user_.item()], reviews_ctr)
                            )
                            
            with torch.no_grad():
                outputs, inter_hidden, inter_attn_score  = InterGRU(interInput, interInput_asin, current_asins, current_reviewerIDs)
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


            loss = torch.sqrt(loss)

            group_loss += loss
            
    num_of_iter = len(training_batches[0])*len(validate_batch_labels)
    RMSE = group_loss/num_of_iter

    return RMSE


if __name__ == "__main__":

    # Doing data preprocessing
    pre_work = Preprocess(opt.setence_max_len, use_nltk_stopword=opt.use_nltk_stopword)
    
    res, itemObj, userObj = pre_work.loadData(sqlfile=opt.sqlfile, testing=False, table= opt.selectTable, rand_seed=opt.train_test_rand_seed)  # for clothing.
    
    # Generate voc & User information
    voc, USER = pre_work.Generate_Voc_User(res, having_interaction=opt.having_interactions)

    # Generate training labels
    training_batch_labels = list()
    candidate_asins = list()
    candidate_reviewerIDs = list()
    label_sen_batch = None

    for idx in range(0, opt.num_of_rating, 1):
        training_labels, training_asins, training_reviewerIDs = pre_work.GenerateLabelEncoding(USER, 
            opt.num_of_reviews+idx, 1, itemObj, userObj)
        
        _batch_labels, _asins, _reviewerIDs, _label_sen_batches = pre_work.GenerateBatchLabelCandidate(training_labels, training_asins, training_reviewerIDs, opt.batchsize,
            USER, itemObj, voc, start_of_reviews=opt.num_of_reviews, num_of_reviews = opt.num_of_rating , testing=False)
        
        training_batch_labels.append(_batch_labels)
        candidate_asins.append(_asins)
        candidate_reviewerIDs.append(_reviewerIDs)

        # Batches of sentences that are LABEL.
        label_sen_batch =_label_sen_batches


    # If loading Pre-train words
    if(opt.use_pretrain_wordVec == 'Y'):
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


    # Generate training batches & Train
    if(opt.mode == "train" or opt.mode == "both"):
        training_batches, training_asin_batches = pre_work.GenerateTrainingBatches(USER, itemObj, voc, num_of_reviews=opt.num_of_reviews, batch_size=opt.batchsize)

        Train(voc, opt.selectTable, training_batches, training_asin_batches, candidate_asins, candidate_reviewerIDs, training_batch_labels, label_sen_batch,
            opt.save_dir, TrainEpoch=opt.epoch, latentK=opt.latentK, hidden_size=opt.hidden, intra_method=opt.intra_attn_method , inter_method=opt.inter_attn_method,
            learning_rate = opt.lr, dropout=opt.dropout, isStoreModel=True, WriteTrainLoss=True, store_every = opt.save_model_freq, 
            use_pretrain_item=False, isCatItemVec=True, pretrain_wordVec=pretrain_wordVec)

    # Generate testing batches
    if(opt.mode == "test" or opt.mode == "showAttn" or 
        opt.mode == "both" or opt.mode == "test_generation"):

        if(opt.test_on_traindata =='Y'):
            test_on_traindata = True
        else:
            test_on_traindata = False

        # Loading testing data from database
        res, itemObj, userObj = pre_work.loadData(sqlfile=opt.sqlfile, testing=True, 
            test_on_traindata=test_on_traindata, table=opt.selectTable, rand_seed=opt.train_test_rand_seed)   # clothing
        USER = pre_work.Generate_Voc_User(res, having_interaction=opt.having_interactions, generateVoc=False)

        # Generate testing labels
        testing_batch_labels = list()
        candidate_asins = list()
        candidate_reviewerIDs = list()
        testing_sen_batches = None

        for idx in range(0, opt.num_of_rating, 1):

            testing_labels, testing_asins, testing_reviewerIDs = pre_work.GenerateLabelEncoding(USER, 
                opt.num_of_reviews+idx, 1, itemObj, userObj)
            
            _batch_labels, _asins, _reviewerIDs, _testing_sen_batches = pre_work.GenerateBatchLabelCandidate(testing_labels, testing_asins, testing_reviewerIDs, opt.batchsize,
                USER, itemObj, voc, start_of_reviews=opt.num_of_reviews, num_of_reviews = opt.num_of_rating , testing=True)

            testing_batch_labels.append(_batch_labels)
            candidate_asins.append(_asins)
            candidate_reviewerIDs.append(_reviewerIDs)
            testing_sen_batches = _testing_sen_batches

        # Generate testing batches
        testing_batches, testing_asin_batches = pre_work.GenerateTrainingBatches(USER, itemObj, voc, 
            num_of_reviews=opt.num_of_reviews, batch_size=opt.batchsize, testing=True)

    # Testing(chose epoch)
    if(opt.mode == "test_generation"):

        # Setup epoch being chosen
        chose_epoch = opt.epoch

        # Loading IntraGRU
        IntraGRU = list()
        for idx in range(opt.num_of_reviews):
            model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, chose_epoch))
            IntraGRU.append(model)

        # Loading InterGRU
        InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, chose_epoch))
        # Loading DecoderModel
        DecoderModel = torch.load(R'{}/Model/DecoderModel_epoch{}'.format(opt.save_dir, chose_epoch))

        # evaluating
        tokens_dict, scores_dict = evaluate(IntraGRU, InterGRU, DecoderModel, testing_batches, testing_asin_batches, testing_batch_labels, candidate_asins, candidate_reviewerIDs, testing_sen_batches,
            isCatItemVec=True, userObj=userObj, itemObj=itemObj, voc=voc)


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

            # Loading DecoderModel
            DecoderModel = torch.load(R'{}/Model/DecoderModel_epoch{}'.format(opt.save_dir, Epoch))

            # evaluating
            RMSE = evaluate_RMSE(IntraGRU, InterGRU, DecoderModel, testing_batches, testing_asin_batches, testing_batch_labels, candidate_asins, candidate_reviewerIDs, testing_sen_batches,
            isCatItemVec=True, userObj=userObj, itemObj=itemObj, voc=voc)
            print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))

            with open(R'{}/Loss/TestingLoss_RMSE.txt'.format(opt.save_dir),'a') as file:
                file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))    



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
            isCatItemVec=True, randomSetup=-1, isWriteAttn=True, userObj=userObj)
