#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
import scipy.io 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier 
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap


from sklearn.model_selection import train_test_split 

train_data = scipy.io.loadmat('train_32x32.mat')


# In[2]:


# extract the images (X) and labels (y) from the dict
#X = data input y = data output(label gambar)
X = train_data['X'] 
y = train_data['y'] 
#data = np.loadtxt('ex2data1.txt',delimiter=',')


# In[3]:


img_index = 80
plt.imshow(X[:,:,:,img_index])
plt.show()
print(y[img_index])


# In[4]:


X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T
y = y.reshape(y.shape[0],)
X= X[:1000]
y= y[:1000]

y_coba = [y[80]]
x_coba = [X[80]]

X, y = shuffle(X, y, random_state=42)
print(x_coba)


# In[5]:


clf = RandomForestClassifier()
print(clf)


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), 
                                                        X_train, 
                                                        y_train,
                                                        # Number of folds in cross-validation
                                                        cv=10,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))


# In[8]:



train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)


test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")


plt.title("Learning Curve Random Forest")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# In[9]:


clf.fit(X_train, y_train)


# In[10]:


predic = clf.predict(x_coba)
predic


# In[11]:


preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,preds))
print(classification_report(y_test,preds))


# In[12]:


dt = tree.DecisionTreeClassifier()
print(dt)


# In[13]:


dt.fit(X_train, y_train)


# In[14]:


nb = GaussianNB()
nb.partial_fit(X_train, y_train, np.unique(y_train))


# In[15]:


#Gaussian Naive Bayes training plot
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(GaussianNB(), 
                                                        X_train, 
                                                        y_train,
                                                        # Number of folds in cross-validation
                                                        cv=10,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))


# In[16]:


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)


test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")


plt.title("Learning Curve Naive Bayes")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# In[17]:


nbpred = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test,nbpred))
print(confusion_matrix(y_test,nbpred))
print(classification_report(y_test,nbpred))


# In[18]:


dtPreds = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test,dtPreds))
print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))


# In[19]:


from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(tree.DecisionTreeClassifier(), 
                                                        X_train, 
                                                        y_train,
                                                        # Number of folds in cross-validation
                                                        cv=10,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))


# In[20]:



train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)


test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")


plt.title("Learning Curve Decision Tree")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# In[ ]:




