
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/dataprogram3/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# read the data from training dataset
trainingDataFrame = pd.read_csv(filepath_or_buffer="../input/dataprogram3/train-1.csv", header=None, sep='delimiter', engine="python")

# separate documents from classes and documents
vals = trainingDataFrame.iloc[:,:].values

trainingValue = []
trainingClass = []
for value in vals:
    trainingClass.append(value[0][:1])
    trainingValue.append(value[0][2:])


# In[ ]:


len(trainingValue)


# In[ ]:


testingDataFrame = pd.read_csv(filepath_or_buffer="../input/dataprogram3/test-1.csv", header=None, sep='delimiter', engine="python")

vals = testingDataFrame.iloc[:,:].values

testingValue = []
for value in vals:
     testingValue.append(value[0][:])


# In[ ]:


len(testingValue)


# In[ ]:


trainVector = TfidfVectorizer(min_df=3, max_features=5000, lowercase=True, 
                              strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                              ngram_range=(1, 2), norm='l2', use_idf=1, smooth_idf=1, sublinear_tf=1,
                              stop_words='english')
X_train = trainVector.fit_transform(trainingValue)


# In[ ]:


def KNN(id, pred = [], k=1):
    
    # merge the test data with training dataset
    testData = []
    testData.append(testingValue[id])
    Y = trainVector.transform(testData)
    trainTestVectorCombined = sp.vstack([X_train, Y], format="csr")

    # classify vector x using kNN and majority vote rule given training data and associated classes
    # find nearest neighbors for x
    x = trainTestVectorCombined[len(trainingValue),:]
    dots = x.dot(trainTestVectorCombined.T)
    dots[0, len(trainingValue)] = -1
    sims = list(zip(dots.indices, dots.data))
    sims.sort(key=lambda x: x[1], reverse=True)
    tc = Counter(trainingClass[s[0]] for s in sims[:k]).most_common(2)
    
    if len(tc) < 2 or tc[0][1] > tc[1][1]:
        # majority vote
        return tc[0][0]
    # tie break
    tc = defaultdict(float)
    for s in sims[:k]:
        tc[trainingClass[s[0]]] += s[1]
    return sorted(tc.items(), key=lambda x: x[1], reverse=True)[0][0]


# In[ ]:


predictions = []
for id in range(len(testingValue)):
    predictions.append(KNN(id, k=7))


# In[ ]:


with open('format.dat.txt', 'w') as f:
    for i in predictions:
        f.write('{}\n'.format(i))

