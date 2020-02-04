#%%
import numpy as np
import io

SOS = np.random.uniform(low=-1, high=1, size=(300))
EOS = np.random.uniform(low=-1, high=1, size=(300))



# %%
SOS, EOS

# %%
SOS.tolist()

# %%
fname = '/home/kdd2080ti/Documents/Sean/RecommendationSystem/HNAGE/data/clothing_festtext_subEmb_.vec'


with open(fname, 'a') as _file:
    tmpStr = ''
    for val in EOS.tolist():
        tmpStr = tmpStr + str(val) + ' '
    _file.write('{} {}\n'.format('EOS', tmpStr))

# %%
