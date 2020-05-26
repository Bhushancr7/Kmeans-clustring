# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:18:08 2020

@author: patel
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\patel\\Downloads\\crime_data_clust.csv") 
data.head() 

data=data.iloc[:,1:]

from sklearn.preprocessing import MinMaxScaler
norm=MinMaxScaler()  
norm.fit(data)
norm_data=norm.transform(data) 

type(norm_data)

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


k = list(range(2,15))

k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(norm_data)
    WSS = [] # variable for storing within sum of squares for each cluster  
    for j in range(i):
       j
       WSS.append(sum(cdist(norm_data[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,norm_data.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))    

plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
plt.show 

#now considering number of clusters =4 as analysed in the elbow plot

model = KMeans(n_clusters = 4)
model.fit(norm_data)

model.labels_
series_array=pd.Series(model.labels_)
data["clust"]=series_array
data.iloc[:,1:4].groupby(data.clust).mean()
data.to_csv("crime_data_clust.csv")
