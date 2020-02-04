#%%
from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
import io


#%%
fname = 'PretrainWord/wiki-news-300d-1M.vec'
model = KeyedVectors.load_word2vec_format(fname, binary=False)

print("Loading complete.")

# #%%
# print(model.most_similar('teacher'))
# %%
from utils.preprocessing import Preprocess

def writeVoc(sqlfile= 'HNAE/SQL/cloth_selectALL.sql', selectTable = 'clothing_'):

    res, itemObj, userObj = pre_work.loadData( sqlfile=sqlfile, testing=False, table= selectTable)  # for clothing.
    # Generate voc 
    voc = pre_work.Generate_Voc(res)
    fname = 'clothing_ALL_Voc.txt'

    with open('HNAE/data/' + fname, 'w') as file:
        for word in voc.word2index:
            file.write('{},'.format(word))

pre_work = Preprocess()

#%%
with open('HNAE/data/clothing_ALL_Voc.txt', 'r') as file:
    content = file.read()

words = content.split(',')


# %%
model.wv[words[50]], words[50]
#%%
for val in model.wv[words[50]]:
    print(float(val))

#%%
import torch
def StoreWordSemantic(words, dim, fname):
    
    with open(fname, 'a') as _file:
        _file.write('{} {}\n'.format(len(words), dim))

    with open(fname, 'a') as _file:
        for word in words:

            try:
                wordVec = model.wv[word]
            except KeyError as msg:
                wordVec = torch.randn(dim)
                           
            tmpStr = ''
            # Each dim val
            for val in wordVec:
                tmpStr = tmpStr + str(float(val)) + ' '        
            
            _file.write('{} {}\n'.format(word, tmpStr))
        
# %%

StoreWordSemantic(words, 300, "HNAE/data/clothing_festtext_subEmb.vec")

# %%
from gensim.models import KeyedVectors

filename = 'HNAE/data/clothing_festtext_subEmb.vec'
model_test = KeyedVectors.load_word2vec_format(filename, binary=False)

model_test.most_similar('great', topn=5)

# %%


# %%
