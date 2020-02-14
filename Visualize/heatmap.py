import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.randn(1,4)

# f, (ax1) = plt.subplots(figsize=(10,1) , nrows=1)
f, (ax1) = plt.subplots()

sns.heatmap(x, annot=True, ax=ax1)



plt.show()