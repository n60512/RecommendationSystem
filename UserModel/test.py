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


asin, reviewerID = Read_Asin_Reviewer()


# %%
len(asin)

# %%
len(reviewerID)

# %%
