## Project 2
### clustered RNA-Sqe data using KMeans clustering and reductional dimension as unsupervised learning techniques with two methods and evaluated the performance on each
- Method 1: applied KMeans directly on both raw data and scaled (normalized) data,
then visualized the clusters using dimension reduction with PCA. I evaluate the
performance using the Adjusted Rand Index and Normalized Mutual Information,
metrics that measure the similarity between clusters rather than the distances within
them. These metrics are suitable for evaluating unsupervised learning, providing an
overall accuracy
- Method 2: performed KMeans on PCA and T-SNE transformation of raw
data, then evaluating KMeans on each reductional space. Since KMeans perform well
on raw data, so for the second part, I transformed raw data to PCA and T-SNE. I
visual comparisons of PCA with true labels and KMeans predict labels, as well as t-
SNE with true labels and KMeans prediction