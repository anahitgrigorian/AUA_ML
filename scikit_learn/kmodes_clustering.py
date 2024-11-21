from kmodes.kmodes import KModes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder

#source: https://www.geeksforgeeks.org/revealing-k-modes-cluster-features-with-scikit-learn/


data = np.array([
    ['red', 'SUV', 'domestic'],
    ['blue', 'sedan', 'imported'],
    ['green', 'SUV', 'imported'],
    ['red', 'sedan', 'domestic'],
    ['blue', 'SUV', 'domestic'],
    ['yellow', 'truck', 'domestic'],
    ['white', 'sedan', 'imported'],
    ['black', 'truck', 'imported'],
    ['green', 'SUV', 'domestic'],
    ['yellow', 'sedan', 'domestic'],
    ['red', 'truck', 'domestic'],
    ['blue', 'SUV', 'imported'],
    ['white', 'SUV', 'domestic'],
    ['black', 'sedan', 'imported'],
    ['yellow', 'truck', 'imported'],
    ['red', 'sedan', 'imported'],
    ['green', 'truck', 'domestic'],
    ['blue', 'SUV', 'domestic'],
    ['white', 'truck', 'imported'],
    ['black', 'SUV', 'imported'],
])

# the Elbow method to find optimal k
costs = []
k_values = range(1, 6)
for k in k_values:
    km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=0)
    km.fit(data)
    costs.append(km.cost_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, costs, marker='o')
plt.title('Elbow Method for K-Modes Clustering')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Cost')
plt.grid()
plt.show()

#applying k-means with optimal k 
optimal_k = 3
km = KModes(n_clusters=optimal_k, init='Huang', n_init=5, verbose=1)
clusters = km.fit_predict(data)

print("\nCluster Assignments:", clusters)
print("\nCentroids of the Clusters:")
for idx, centroid in enumerate(km.cluster_centroids_):
    print(f"Cluster {idx + 1}: {centroid}")

encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(data)
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_encoded)

cluster_colors = ['red', 'blue', 'green']
plt.figure(figsize=(8, 6))
for i, color in enumerate(cluster_colors):
    cluster_points = data_2d[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, label=f"Cluster {i+1}")

centroids_2d = pca.transform(encoder.transform(km.cluster_centroids_))
plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', marker='X', s=200, label='Centroids')

plt.title('K-Modes Clustering Visualization (2D Projection)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid()
plt.show()