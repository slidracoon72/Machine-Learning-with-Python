import random 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs

cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()

#dropping Address as it is categorical
df = cust_df.drop('Address', axis=1)

# Normalising data
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
# convert nan to 0 
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)

clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_

# add labels to data frame (add new column about position of a data point to a particular cluster)
df["Clus_km"] = labels

avg = df.groupby('Clus_km').mean()

# distribution of customers based on their age and income:
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

# 3d plotting
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))