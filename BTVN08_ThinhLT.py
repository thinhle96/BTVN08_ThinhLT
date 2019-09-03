#!/usr/bin/env python
# coding: utf-8

# In[77]:


import re
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from sklearn.linear_model import LinearRegression

from pandas import DataFrame, read_csv
from scipy.stats.stats import pearsonr


# In[78]:


bottle = pd.read_csv("Documents/bottle.csv", usecols=['Depthm', 'T_degC', 'Salnty'])
bottle.head(1000)
bottle.fillna(method='ffill',inplace=True)
bottle.dropna(inplace=True)


# In[79]:


parameters = ['T_degC', 'Depthm']
objective = ['Salnty']

x_real0 = bottle[parameters[0]]
x_real1 = bottle[parameters[1]]
y_real = bottle[objective[0]]


# In[80]:


plt.figure()
plt.scatter(x_real0, y_real)
plt.xlabel(parameters[0])
plt.ylabel(objective[0])

plt.figure()
plt.scatter(x_real1, y_real)
plt.xlabel(parameters[1])
plt.ylabel(objective[0])


# In[81]:


x0 = np.array(x_real0)
x1 = np.array(x_real1)
y1 = np.array(y_real)
print(x0)
print(x1)
print('---------------------------------')
print(y1)
X = np.array([x0,x1]).T
y = np.array([y1]).T


# In[95]:


fig = plt.figure()
one = np.ones((X.shape[0],1))
xbar = np.concatenate((one,X), axis =1)
 
A = np.dot(xbar.T,xbar)
b = np.dot(xbar.T,y)
w = np.dot(np.linalg.pinv(A),b)
print('w=', w)

 
w_0 = w[0][0]
w_1 = w[1][0]
w_2 = w[2][0]

y0= w_0 + w_1*x0 + w_2*x1

def plot_synthetic(model):
    ax= plt.axes(projection ='3d')
    ax.plot3D(x_real0,x_real1,y_real,'bo')
    x0=np.linspace(min(x_real0),max(x_real0),20)
    x1=np.linspace(min(x_real1),max(x_real1),20)
    for x in range(20):
        x1n=np.linspace(x1[x], x1[x], 20)
        ax.plot3D(x0,x1n,model.predict(x0,x1n),'blue')
    for x in range(20):
        x0n=np.linspace(x0[x], x0[x],20)
        ax.plot3D(x0n,x1,model.predict(x0n,x1),'blue')
    ax.set_xlabel("Depth(m)")
    ax.set_ylabel("TdegC")
    ax.set_zlabel("Salty")
    plt.show()
        

model=LinearRegression()
model.fit(X,y)
plot_synthetic(model)


# In[ ]:





# In[ ]:




