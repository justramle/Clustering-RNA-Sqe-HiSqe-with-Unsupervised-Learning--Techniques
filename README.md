
# Clustered RNA-Sqe data using unsupervised learning techniques

### KMeans clustering and reductional dimension with two methods

- Method 1:
  - Applied KMeans directly on both raw data and scaled (normalized) data
    
  - Visualized the clusters using dimension reduction with PCA.
    
  - Ealuating the performance using the Adjusted Rand Index and Normalized Mutual Information,
metrics that measure the similarity between clusters rather than the distances within
them. These metrics are suitable for evaluating unsupervised learning, providing an
overall accuracy.

- Method 2:
  - Transformed raw data to reductional dimension using PCA and T-SNE
    
  - Performed KMeans on PCA and T-SNE
 
  - Visualizing to compare of PCA/T-SNE with true labels and KMeans with predict labels
    
  - Evaluating performance of KMeans on each reductional space with NMI and ARI metrics

### Evaluated the performance

|  | Kmeans on Raw Data | Kmeans on Scaled Data | KMeans on PCA | KMeans on T-SNE |
| ---  | --- | --- | --- | --- |
| ARI | 98.51% | 80% | 80.71% | 99.62% |
| NMI | 97.72%  | 85.62% | 84.18% | 99.47% |


