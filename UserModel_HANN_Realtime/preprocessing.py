#%%
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import re
import unicodedata
import itertools
import time
from DBconnector import DBConnection
import tqdm

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

#%%
class Preprocess:
    def __init__(self):
        pass

    #%% convert all letters to lowercase 
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def indexesFromSentence(self, voc, sentence , MAX_LENGTH = 200):
        sentence_segment = sentence.split(' ')[:MAX_LENGTH]
        return [voc.word2index[word] for word in sentence_segment]

    def indexesFromSentence_Evaluate(self, voc, sentence , MAX_LENGTH = 200):
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

    def zeroPadding(self, l, fillvalue=PAD_token):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    # Returns padded input sequence tensor and lengths
    def inputVar(self, l, voc, testing=False):
        if(testing):
            indexes_batch = [self.indexesFromSentence_Evaluate(voc, sentence) for sentence in l]
        else:   # for training
            indexes_batch = [self.indexesFromSentence(voc, sentence) for sentence in l]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        padList = self.zeroPadding(indexes_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    #%%
    def Read_Asin_Reviewer(self, talbe=''):
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
    def loadData(self, havingCount = 15, LIMIT=5000, testing=False, table='', withOutTable='', through_table=False):

        print('Loading asin/reviewerID from cav file...')
        asin, reviewerID = self.Read_Asin_Reviewer(table)
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
            if(through_table):
                sql = (
                    'SELECT clothing_realtime8_interaction8.*, clothing_review.`asin`, '+
                    'clothing_review.overall, clothing_review.reviewText, clothing_review.unixReviewTime '+
                    'FROM clothing_realtime8_interaction8, clothing_review '+
                    'WHERE clothing_realtime8_interaction8.`ID`=clothing_review.`ID` '+
                    'AND `No` <=10000 '+
                    'ORDER BY `No` ;'                    
                )
            else:
                sql = (
                    'SELECT '+
                    'RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, '+
                    'clothing_review.`ID`, clothing_review.reviewerID , clothing_review.`asin`, '+
                    'clothing_review.overall, clothing_review.reviewText, clothing_review.unixReviewTime '+
                    'FROM clothing_review, clothing_realtime_at6 '+
                    'WHERE clothing_review.reviewerID = clothing_realtime_at6.reviewerID '+
                    'AND clothing_review.unixReviewTime = clothing_realtime_at6.unixReviewTime '+
                    'AND clothing_review.reviewerID IN (SELECT reviewerID FROM clothing_realtime_at6_training_1000)  '+
                    'ORDER BY reviewerID,unixReviewTime ASC ;'
                )

        else:
            if(through_table):
                sql = (
                    'SELECT clothing_realtime8_interaction8.*, clothing_review.`asin`, '+
                    'clothing_review.overall, clothing_review.reviewText, clothing_review.unixReviewTime '+
                    'FROM clothing_realtime8_interaction8, clothing_review '+
                    'WHERE clothing_realtime8_interaction8.`ID`=clothing_review.`ID` '+
                    'AND `No` <=22000 '+
                    'AND `No` >20000 '+
                    'ORDER BY `No` ;'                    
                )
            else:            
                sql = (
                    'SELECT '+
                    'RANK() OVER (PARTITION BY reviewerID ORDER BY unixReviewTime,ID ASC) AS rank, '+
                    'clothing_review.`ID`, clothing_review.reviewerID , clothing_review.`asin`, '+
                    'clothing_review.overall, clothing_review.reviewText, clothing_review.unixReviewTime '+
                    'FROM clothing_review, clothing_realtime_at6 '+
                    'WHERE clothing_review.reviewerID = clothing_realtime_at6.reviewerID '+
                    'AND clothing_review.unixReviewTime = clothing_realtime_at6.unixReviewTime '+
                    'AND clothing_review.reviewerID IN (SELECT reviewerID FROM clothing_realtime_at6_testing_200)  '+
                    'ORDER BY reviewerID,unixReviewTime ASC ;'
                )

        print(sql)

        conn = DBConnection()
        res = conn.selection(sql)
        conn.close()

        print('Loading complete. [{}]'.format(time.time()-st))
        
        return res, itemObj, userObj

    # %%
    def Generate_Voc_User(self, res, havingCount=10, limit_user=1000, generateVoc=True, user_based=False):
        
        USER = list()
        LAST_USER = ''
        LAST_NO = ''
        ctr = -1
        
        # Creating Voc
        st = time.time()
        print('Creating Voc ...') 
        if(generateVoc):
            myVoc = Voc('Review')

        for index in tqdm.tqdm(range(len(res))):
            # Original : user based
            if(user_based):
                # Check is the next user or not
                if(LAST_USER != res[index]['reviewerID'] and len(USER)<limit_user):
                    LAST_USER = res[index]['reviewerID']
                    USER.append(PersonalHistory(res[index]['reviewerID']))
                    ctr += 1   # add counter if change the user id
            else:
                # Check is the next 'Row' or not
                if(res[index]['rank'] == 1):
                    USER.append(PersonalHistory(res[index]['reviewerID']))
                    ctr += 1   # add counter if change the user id                
                pass
            
            if(res[index]['rank'] < havingCount + 1):
                current_sentence = self.normalizeString(res[index]['reviewText'])
                
                if(generateVoc):
                    myVoc.addSentence(current_sentence) # myVoc add word 
                
                if(len(USER)<limit_user+1):
                    USER[ctr].addData(
                                current_sentence,
                                res[index]['overall'], 
                                res[index]['asin'],
                                res[index]['reviewerID']
                            )
                
                tmp = res[index]['No'] 
                USER[ctr]
                stop=1

        print('User length:[{}]'.format(len(USER)))
        print('Voc creation complete. [{}]'.format(time.time()-st))

        if(generateVoc):
            return myVoc, USER
        else:
            return USER

    #%%
    def batch2TrainData(self, myVoc, sentences, rating, isSort = False, normalizeRating = False, testing=False):
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

        inp, lengths = self.inputVar(
            sentences_batch,
            myVoc,
            testing)
        
        label = torch.tensor([val for val in rating_batch])
        # Wheather normalize rating
        if(normalizeRating):
            label = (label-1)/(5-1)

        return inp, lengths, label

    #%%
    def batch2LabelData(self, rating, this_asin, this_reviewerID, itemObj, userObj, normalizeRating = False):
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
    def GenerateTrainingBatches(self, USER, itemObj, voc, num_of_reviews = 5 , batch_size = 5, testing=False):
        new_training_batches_sentences = list()
        new_training_batches_ratings = list()
        new_training_batches = list()
        new_training_batches_asins = list()
        num_of_batch_group = 0
        

        # for each user numberth reivews
        for review_ctr in range(0, num_of_reviews, 1):
            new_training_batch_sen = dict() # 40 (0~39)
            new_training_batch_rating = dict()
            new_training_batch_asin = dict()
            training_batches = dict()
            
            for user_ctr in range(len(USER)):
                
                # Insert group encodeing
                if((user_ctr % batch_size == 0) and user_ctr>0):
                    num_of_batch_group+=1
                    
                    # encode pre group
                    training_batch = self.batch2TrainData(
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
                # asin
                this_user_asin = USER[user_ctr].this_asin[review_ctr]
                this_user_asin_index = itemObj.asin2index[this_user_asin]

                if(num_of_batch_group not in new_training_batch_sen):
                    new_training_batch_sen[num_of_batch_group] = []
                    new_training_batch_rating[num_of_batch_group] = []
                    new_training_batch_asin[num_of_batch_group] = []
                    # training_batches[num_of_batch_group] = []

                new_training_batch_sen[num_of_batch_group].append(this_user_sentence)
                new_training_batch_rating[num_of_batch_group].append(this_user_rating)   
                new_training_batch_asin[num_of_batch_group].append(this_user_asin_index) 

                # Insert group encodeing (For Last group)
                if(user_ctr == (len(USER)-1)):
                    num_of_batch_group+=1
                    # encode pre group
                    training_batch = self.batch2TrainData(
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
            new_training_batches_asins.append(new_training_batch_asin)

            num_of_batch_group = 0
        return new_training_batches, new_training_batches_asins

    #%%
    def GenerateLabelEncoding(self, USER, num_of_reviews, num_of_rating, itemObj, userObj):
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

            new_training_label, asin_batch, reviewerID_batch = self.batch2LabelData(new_training_label, new_training_asin, new_training_reviewerID, itemObj, userObj)
            
            training_labels[user_ctr] = new_training_label   
            training_asins[user_ctr] = asin_batch   
            training_reviewerIDs[user_ctr] = reviewerID_batch  

        return training_labels, training_asins, training_reviewerIDs
        
    #%%
    def GenerateBatchLabelCandidate(self, labels_, asins_, reviewerIDs_, batch_size):
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
