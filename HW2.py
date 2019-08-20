
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from scipy.sparse import csr_matrix, find
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import random
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.utils import shuffle
from sklearn.metrics import calinski_harabaz_score


# In[2]:


file = 'C:/Users/akash/Documents/train.dat'
data = open(file, 'r')
# print data


# In[3]:


docs = list()
for row in data:
    docs.append(row.rstrip().split(" "))


# In[4]:


# sperate indices and values
dataIndex = list()
value = list()
for d in docs:
    d_index = list()
    d_value = list()
    for i in range(0,len(d),2):      
        d_index.append(d[i])
    for j in range(1,len(d),2):     
        d_value.append(d[j])
    dataIndex.append(d_index)
    value.append(d_value)


# In[5]:


nrows = len(docs)

idx = {}
tid = 0
nnz = 0
ncol = 0
_max = list()
for d in dataIndex:
    nnz += len(d)
    _max.append(max(d))
    for w in d:
        if w not in idx:
            idx[w] = tid
            tid += 1


# In[6]:


def building_CR(dataIndex, value):
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0
    n = 0
    
    for (d,v) in zip(dataIndex, value):
        l = len(d)
        for j in range(l):
            ind[int(j) + n] = d[j]
            val[int(j) + n] = v[j]
        
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
    
    matrix_form = csr_matrix((val, ind, ptr), shape=(nrows, max(ind)+1), dtype=np.double)
    matrix_form.sort_indices()
    
    return matrix_form


# In[7]:


matrix_form = building_CR(dataIndex, value)


# In[8]:


# scale matrix and normalize its rows
def scale_and_normalize(matrix_form, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        matrix_form = matrix_form.copy()
    nrows = matrix_form.shape[0]
    nnz = matrix_form.nnz
    ind, val, ptr = matrix_form.indices, matrix_form.data, matrix_form.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else matrix_form

def CSRrowmatrix(matrix_form, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        matrix_form = matrix_form.copy()
    nrows = matrix_form.shape[0]
    nnz = matrix_form.nnz
    ind, val, ptr = matrix_form.indices, matrix_form.data, matrix_form.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = float(1.0/np.sqrt(rsum))
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return matrix_form


# In[9]:


matrix_form2 = scale_and_normalize(matrix_form, copy=True)
matrix_form3 = CSRrowmatrix(matrix_form2, copy=True)


# In[11]:


print (matrix_form3.shape)


# In[12]:


def initial_cent(x, k):
    x_shuffle = shuffle(x, random_state=0)
    return x_shuffle[:k,:]


# In[13]:


def sumil(x1, x2):
    sumils = x1.dot(x2.T)
    return sumils


# In[14]:


def findCentroids(matrix_form, centroids):
    idx = list()
    sumilsMatrix = sumil(matrix_form, centroids)

    for i in range(sumilsMatrix.shape[0]):
        row = sumilsMatrix.getrow(i).toarray()[0].ravel()
        top_indices = row.argsort()[-1]
        top_values = row[row.argsort()[-1]]
        idx.append(top_indices + 1)
    return idx


# In[15]:


def computeMeans(matrix_form, idx, k):
    centroids = list()
    for i in range(1,k+1):
        indi = [j for j, x in enumerate(idx) if x == i]
        members = matrix_form[indi,:]
        if (members.shape[0] > 1):
            centroids.append(members.toarray().mean(0))
    centroids_csr = csr_matrix(centroids)
    return centroids_csr


# In[16]:


def bisect_k_means(k, matrix_form, n_iter):
    centroids = initial_cent(matrix_form, k)
    for _ in range(n_iter): 
        idx = findCentroids(matrix_form, centroids)
        centroids = computeMeans(matrix_form, idx, k)
    return idx


# In[18]:


x_axis = list()
y_axis = list()
for k in range(3, 22, 2):
    idx = bisect_k_means(k, matrix_form3, 10)
    score = calinski_harabaz_score(matrix_form3.toarray(), idx)
    print (k, score)
    x_axis.append(k)
    y_axis.append(score)


# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


plt.plot(x_axis, y_axis)


# In[23]:


print ("Score: ")
print(calinski_harabaz_score(matrix_form3.toarray(), idx))


# In[24]:


# print result to text file
text_file = open("format.dat", "w")
for i in idx:
    
    text_file.write(str(i) +'\n')
text_file.close()

