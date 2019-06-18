#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


file_path=''


# In[4]:


import glob
import os
file_list = glob.glob(os.path.join(file_path, "*.txt"), recursive=True)


# In[5]:


for file in file_list:
    print(file)


# In[6]:


col_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak',
             'slope','ca','thal','target']


# In[7]:


df_cleveland = pd.read_csv("pro_cleveland.txt", 
                           names=col_names, header=None, na_values="?")


# In[8]:


df_switzerland = pd.read_csv("pro_switzerland.txt", 
                             names=col_names, header=None, na_values="?")


# In[9]:


df_va = pd.read_csv("pro_va.txt", 
             
                    names=col_names, header=None, na_values="?")


# In[10]:


df_va


# In[11]:


new_df=df_va.append(df_switzerland,ignore_index=True)


# In[12]:


new_df


# In[13]:


df_hungarian = pd.read_csv("pro_hungarian.txt", 
                           names=col_names, header=None, na_values="?")


# In[14]:


new_df1=new_df.append(df_hungarian,ignore_index=True)


# In[15]:


new_df1


# In[16]:


heart_disease_df=new_df1.append(df_cleveland,ignore_index=True)


# In[17]:


heart_disease_df


# In[64]:


heart_disease_df=df.drop(['ca','thal'], axis=1)


# In[65]:


heart_disease_df


# In[66]:


from sklearn.preprocessing import Imputer
import   math
imput = Imputer(missing_values='NaN',strategy='mean')
heart_disease_df = list(imput.fit_transform(heart_disease_df))

for i in range(919):
    for j in range(12):
        heart_disease_df[i][j] = math.ceil(heart_disease_df[i][j])


# In[68]:


df = pd.DataFrame(heart_disease_df)

df.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak',
             'slope','target']
           


# In[69]:


df


# In[70]:


number=[1,2,3,4]
for col in df.itertuples():

    if col.cp in number:
        df['target'].replace(to_replace=col.cp, value=1, inplace=True)


# In[71]:


X = df.drop('target',axis=1)
y = df['target'].values


# In[72]:


from sklearn.model_selection import train_test_split


# In[73]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 


# In[74]:


from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)  


# In[75]:


y_pred = svclassifier.predict(X_test) 


# In[76]:


from sklearn import metrics


# In[77]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[78]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[79]:


from sklearn.metrics import confusion_matrix


# In[80]:


print('---------------Confusion Matrix---------')
print(confusion_matrix(y_test, y_pred))  


# In[81]:


from sklearn.metrics import confusion_matrix
import seaborn as sn
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# In[82]:


from sklearn.metrics import classification_report
print('---------------Precision Scores---------')
print(classification_report(y_test, y_pred))  


# In[ ]:





# In[ ]:




