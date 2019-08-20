
# coding: utf-8

# In[8]:


# Import libraries
import numpy as np
import pandas as pd

# Import libraries from Surprise package
from surprise import Reader, Dataset, SVD, evaluate

# Read train data
trainData = pd.read_csv('train.dat', sep='\t', encoding='latin-1', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Read test data
testData = pd.read_csv('test.dat', sep='\t', encoding='latin-1', names=['user_id', 'item_id'])

# Get the length of test data 
testDataLength = testData.user_id.count()

# Separate user_id from item_id
vals = testData.iloc[:,:].values

testUserID = []
testItemID = []
for value in vals:
    testUserID.append(value[0])
    testItemID.append(value[1])

# Load Reader library
reader = Reader()

# Load ratings dataset with Dataset library
data = Dataset.load_from_df(trainData[['user_id', 'item_id', 'rating']], reader)

# Split the dataset for 10-fold evaluation
data.split(n_folds=10)


# In[9]:


# Use the SVD algorithm.
svd = SVD()


# In[10]:


trainset = data.build_full_trainset()
svd.train(trainset)


# In[11]:


length = testData.user_id.count()

with open('format.dat.txt', 'w') as f:
    for i in range(testDataLength):
        result = svd.predict(testUserID[i], testItemID[i])
        f.write('{}\n'.format(round(result.est)))

