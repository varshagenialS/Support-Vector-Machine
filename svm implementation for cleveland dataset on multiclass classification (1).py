#!/usr/bin/env python
# coding: utf-8

# In[28]:


import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[29]:


file_path=''


# In[30]:


import glob
import os
file_list = glob.glob(os.path.join(file_path, "*.txt"), recursive=True)


# In[31]:


for file in file_list:
    print(file)


# In[32]:


col_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak',
             'slope','ca','thal','target']


# In[33]:


df_cleveland = pd.read_csv("pro_cleveland.txt", 
                           names=col_names, header=None, na_values="?")


# In[34]:


df_cleveland


# In[35]:


from sklearn.preprocessing import Imputer
import   math
imput = Imputer(missing_values='NaN',strategy='mean')
df_cleveland = list(imput.fit_transform(df_cleveland))

for i in range(302):
    for j in range(14):
        df_cleveland[i][j] = math.ceil(df_cleveland[i][j])


# In[36]:


df = pd.DataFrame(df_cleveland)

df.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak',
             'slope','ca','thal','target']
           


# In[37]:


df


# In[47]:


X = df.drop('target',axis=1)
y = df['target'].values


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 


# In[50]:


from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train) 


# In[51]:


y_pred = svclassifier.predict(X_test) 


# In[52]:


from sklearn import metrics


# In[53]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[54]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, pos_label=1, average=None))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, pos_label=1, average=None))


# In[55]:


from sklearn.metrics import confusion_matrix
import seaborn as sn
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1','Predicted:2','Predicted:3','Predicted:4'],index=['Actual:0','Actual:1','Actual:2','Actual:3','Actual:4'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# In[ ]:





# In[ ]:




