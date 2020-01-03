#1219_clothing_pre_cat_r4_bs40_lr5e05_lk32_dec20_dp0_interGeneral


#%%
from gensim.models import KeyedVectors
import sys

# %%
filename = 'torchWordEmbedding/WordSemantic_0101'
model = KeyedVectors.load_word2vec_format(filename, binary=False)


# %%
# model.most_similar('woman')
model.most_similar('comfortable', topn=20)

# %%
model.similarity('good','nice')
#%%


# %%

while(True):
    print('\n=======================================')
    print('1) Calculate similarity. (e.g. word1 word2)')
    print('2) Search topk. (e.g. word1 k)')
    
    ch = input('Your choice:')
    
    if(ch == '1'):
        input_text = input().split()
        print(model.similarity(input_text[0], input_text[1]))
    elif(ch == '2'):
        input_text = input().split()
        # print(input_text[0], int(input_text[1]))
        for val in (model.most_similar(input_text[0], topn=int(input_text[1]))):
            print(val)
    else:
        break