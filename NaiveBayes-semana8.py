
# coding: utf-8

# # Gaussian Naive Bayes

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# The aim is to construct a classifier that predicts whether a user will buy a new SUV given information of his/her Age and Salary.
# 

# In[2]:



dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


# ## Splitting the dataset into the Training set and Test set

# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# ## Feature Scaling

# In[4]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[5]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# ## Predicting the Test set results

# In[6]:


y_pred = classifier.predict(X_test)


# ## Making the Confusion Matrix

# In[7]:



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# ## Visualising the Training set results

# In[8]:


# larger graph 
width = 15
height = 15
plt.figure(figsize=(width, height))
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# ## Visualising the Test set results

# In[9]:


# larger graph 
width = 15
height = 15
plt.figure(figsize=(width, height))
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
#%%

# ## Excercise: Try your own Gaussian Naive Bayes algorithm 

# - Hint: compute the mean and standard deviation for each class value and variable|class
# 
# - Use a Gaussian function to estimate the probability of a specific attribute value. To do so, you can use the mean and standard deviation computed for that attribute from the training data.
# 
# - You should obtain the same confusion matrix of sklearn algorithm
# 

#%%
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#%%
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
#%%
num_class = 2
means= np.zeros((2,np.size(X_train,1)))
SDs =  np.zeros((2,np.size(X_train,1)))
p_c = np.zeros((num_class))
sum = 0
predicts = np.zeros((np.size(y_test,0)))

def calculate_mean():
    for c in range (0,num_class):
        for num_feature in range (0,np.size(X_train,1)):
            sum_c = 0
            count_c = 0
            for i in range(0,np.size(X_train,0)):
                if c == y_train[i]:
                    count_c += 1 
                    sum_c += X_train[i,num_feature]
            means[c,num_feature] = sum_c / count_c
    
def calculate_SD():
    for c in range (0,num_class):
        for num_feature in range (0,np.size(X_train,1)):
            sum_c = 0
            count_c = 0
            for i in range(0,np.size(X_train,0)):
                if c == y_train[i]:
                    sum_c += np.power(X_train[i,num_feature] - means[c,num_feature],2)
                    count_c += 1
            SD = 1.0/count_c * sum_c 
            SDs[c,num_feature] = np.sqrt(SD) 

def calculate_pc():
    for i in range (0,num_class):
        count = 0
        for j in range (0,np.size(y_train,0)):
            if i == y_train[j]:
                count += 1
        print(count)
        p_c[i] = float(count) / float(np.size(y_train,0))
        
def distribution_gaussian(x,c,num_feature):
    return (1.0/ (np.sqrt(2.0 * np.pi * (SDs[c,num_feature]**2))) * (np.exp(-1*(np.power(x-means[c,num_feature],2)/(2*np.power(SDs[c,num_feature],2))))))
        
def run():
    #calculate means for each feature 
    calculate_mean()
    
    #calculate SD for each feature 
    calculate_SD()
    
    #calculate previous case 
    calculate_pc()
    
    #predicting 
    count = 0
    
    for x in X_test:
        predict_class_1 = 0.
        predict_class_2 = 0.
        acu_class1 = 1
        acu_class2 = 1
        for num_feature in range(0,np.size(X_test,1)):
            #print(x[num_feature])
            acu_class1 *= distribution_gaussian(x[num_feature],0,num_feature)
            
        for num_feature in range(0,np.size(X_test,1)):
            #print(x[num_feature])
            acu_class2 *= distribution_gaussian(x[num_feature],1,num_feature)
        
        p_class_1 = p_c[0] * acu_class1  
        p_class_2 = p_c[1] * acu_class2
        predict_class_1 = p_class_1 / (p_class_1 + p_class_2)
        predict_class_2 = p_class_2 / (p_class_1 + p_class_2)
        #print("class 1",predict_class_1)
        #print("class 2",predict_class_2)
        if(predict_class_1 > predict_class_2):
            predicts[count] = 0
        else:
            predicts[count] = 1
        count+=1
        
    #calculate accuracy 
    cm = confusion_matrix(y_test, predicts)
    accuracy = np.mean(y_test==predicts)
    #RESULTADOS
    print("Results with my own Naive Bayes")
    print(accuracy)
    print(cm)
    
#exe naive bayes 
run()