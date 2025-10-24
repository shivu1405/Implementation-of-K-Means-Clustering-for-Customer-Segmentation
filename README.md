# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. K-Means Clustering

2. Hierarchical Clustering

3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

4. Gaussian Mixture Model (GMM)


## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:  Shivasri
RegisterNumber:  212224220098


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df= pd.read_csv("Mall_Customers.csv")

df.head()
df.info()
df.isnull().sum()

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init="k-means++",n_init=10)
  kmeans.fit(df.iloc[:,3:])
  wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.xlabel("No of clusters")
plt.ylabel("wcss")
plt.title("Elbow method")
km=KMeans(n_clusters=5,n_init=10)

km.fit(df.iloc[:,3:])
y_pred=km.predict(df.iloc[:,3:])
y_pred

df["cluster"]=y_pred
dt0=df[df["cluster"]==0]
dt1=df[df["cluster"]==1]
dt2=df[df["cluster"]==2]
dt3=df[df["cluster"]==3]
dt4=df[df["cluster"]==4]
plt.scatter(dt0["Annual Income (k$)"],dt0["Spending Score (1-100)"],c="red",label="cluster1")
plt.scatter(dt1["Annual Income (k$)"],dt1["Spending Score (1-100)"],c="black",label="cluster2")
plt.scatter(dt2["Annual Income (k$)"],dt2["Spending Score (1-100)"],c="blue",label="cluster3")
plt.scatter(dt3["Annual Income (k$)"],dt3["Spending Score (1-100)"],c="green",label="cluster4")
plt.scatter(dt4["Annual Income (k$)"],dt4["Spending Score (1-100)"],c="magenta",label="cluster5")
plt.legend()
plt.title("Customer Segments")

*/
```

## Output:
<img width="1411" height="688" alt="image" src="https://github.com/user-attachments/assets/2205c092-7fd9-4c30-8dd0-d56a07c97c89" />
<img width="1409" height="680" alt="image" src="https://github.com/user-attachments/assets/f20f2355-0c27-44fe-8891-e8e8b8e6f60b" />

<img width="1574" height="718" alt="image" src="https://github.com/user-attachments/assets/1ed48026-100c-4c24-b15a-1020fd38b989" />

<img width="1281" height="284" alt="image" src="https://github.com/user-attachments/assets/a823e5dc-7d94-40c9-8a4d-93a90c4fe61c" />


<img width="1611" height="731" alt="image" src="https://github.com/user-attachments/assets/68896d6b-74c5-470f-bd17-eadf35bb6ac3" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
