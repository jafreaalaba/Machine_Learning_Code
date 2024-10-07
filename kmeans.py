import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(40)
X = np.random.rand(100, 2)

#Create two clusters
X[:50] += 1
X[50:] += 2

#APply KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

#Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

#Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0],X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='X', label='centroids', s=200)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()