#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble
import warnings
warnings.filterwarnings('ignore')


# In[5]:


dataframe=pd.read_csv("heart.csv")
dataframe.head(10)


# In[6]:


dataframe.info()


# In[7]:


dataframe.isna().sum()


# In[8]:


plt.figure(figsize=(15,10))
sns.heatmap(dataframe.corr(),linewidth=.01,annot=True, cmap="summer")
plt.show()
plt.savefig('Correlationfigure')


# In[9]:


dataframe.hist(figsize=(12,12))
plt.savefig('featuresplot')


# In[10]:


X=dataframe.drop(['target'],axis=1)
y=dataframe['target']


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=40)


# In[12]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
tree_model=DecisionTreeClassifier(max_depth=8,criterion='entropy')
cv_scores=cross_val_score(tree_model,X,y,cv=10,scoring='accuracy')
m=tree_model.fit(X_train,y_train)
prediction=m.predict(X_test)
cm=confusion_matrix(y_test,prediction)
sns.heatmap(cm,annot=True,cmap='winter',linewidths=0.3,linecolor='Black',annot_kws={"size":20})
print(classification_report(y_test,prediction))
TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]
print('Testing accuracy:',(TP+TN)/(TP+TN+FN+FP))
print('Testing sensitivity or Recall:',TP/(TP+FN))
print('Testing specificity:',TN/(TN+FP))
print('Testing precision:',TP/(TP+FP))


# In[13]:


input=(63,1,3,145,233,1,0,150,0,2.3,0,0,1)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=tree_model.predict(input_reshaped)
if(pre1==1):
    print("Patient has heart disease")
else:
    print("Patient is healthy")


# In[14]:


print(dataframe.columns)


# In[15]:


input=(39,0,1,154,249,1,1,158,0,2.9,2,0,4)
input_as_numpy=np.asarray(input)
input_reshaped=input_as_numpy.reshape(1,-1)
pre1=tree_model.predict(input_reshaped)
if(pre1==1):
    print("Patient has heart disease")
else:
    print("Patient is healthy")

