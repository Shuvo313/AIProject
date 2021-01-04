#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import matplotlib
import numpy
import pandas
import scipy
import sklearn

print("Python:",sys.version)
print("Matplotlib",matplotlib.__version__)
print("Numpy:",numpy.__version__)
print("Pandas:",pandas.__version__)
print("Scipy:",scipy.__version__)
print("Sklearn:",sklearn.__version__)


# In[2]:


from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[3]:


url="Datasets/IrisFlower/Diabetes.csv"
names= ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
dataset= pandas.read_csv(url,names=names)


# In[4]:


dataset.shape


# In[5]:


dataset.head(20)


# In[6]:


dataset.describe()


# In[7]:


dataset.hist()
plt.show()


# In[8]:


scatter_matrix(dataset)
plt.show()


# In[9]:


dataset.groupby('Outcome').size()


# In[10]:


array=dataset.values
X=array[:,0:8]
Y=array[:,8]
validation=0.20
seed=7
X_train,X_validation,Y_train,Y_validation=model_selection.train_test_split(X,Y,test_size=validation,random_state=seed)


# In[11]:


scoring='accuracy'


# In[12]:


models=[]
models.append(('LR:',LogisticRegression()))
models.append(('KNN:',KNeighborsClassifier()))
models.append(('SVC:',SVC()))


# In[13]:


results=[]
names=[]

for name,model in models:
    kfold = model_selection.KFold(n_splits=10,random_state = seed)
    cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(name," :Mean = ",cv_results.mean()," :Std =",cv_results.std())


# In[14]:


for name,model in models:
    model.fit(X_train,Y_train)
    predictions=model.predict(X_validation)
    print(name)
    print(accuracy_score(Y_validation,predictions)*100)
    print(classification_report(Y_validation,predictions))


# In[ ]:




