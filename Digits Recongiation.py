#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
digits = load_digits()


# In[3]:


plt.gray() 
for i in range(3):
    plt.matshow(digits.images[i])


# In[7]:


digits.target


# In[8]:


dir(digits)


# In[9]:


digits.target_names


# In[10]:


df = pd.DataFrame(digits.data,digits.target)
df


# In[14]:


df['target'] = digits.target
df.head(20)


# # Training and Testing model

# In[59]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.5)


# In[60]:


from sklearn.svm import SVC
rbf_model= SVC(kernel='rbf')


# In[61]:


len(X_train)


# In[53]:


len(X_test)


# # Fitting the model

# In[62]:


rbf_model.fit(X_train, y_train)


# In[63]:


rbf_model.score(X_test,y_test)


# In[64]:


print('Accuracy =',rbf_model.score(X_test,y_test))


# # Using linear kernel

# In[65]:


linear_model= SVC(kernel='linear')
linear_model.fit(X_train,y_train)


# In[67]:


linear_model.score(X_test,y_test)


# In[68]:


print('Accuracy =',linear_model.score(X_test,y_test))


# In[72]:


from sklearn.externals import joblib


# In[ ]:




