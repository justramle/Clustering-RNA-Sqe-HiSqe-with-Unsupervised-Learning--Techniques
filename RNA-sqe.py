import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ==========Method 1===========
# Load Data
data = pd.read_csv('data-1.csv')
y_true = pd.read_csv('labels.csv')

# drop the first columns features in samples because it's string
data = data.drop(columns=["Unnamed: 0"])
# drop 1st columns in labels
y_true = y_true.drop(columns=["Unnamed: 0"])

# Standardizing data
scaler = np.asarray(data)
gene_data_scaled = StandardScaler()

# # fit and transform data 
gene_data_scaled = gene_data_scaled.fit_transform(scaler)

# fit data into kmeans
kmeans = KMeans(n_clusters=5, n_init=50, random_state=50)
# return  cluster labels for each data points
y_predict = kmeans.fit_predict(data)  


# encoding the labels into numerical
label_encoder = LabelEncoder()
y_true_numerical = label_encoder.fit_transform(y_true['Class'])

# Evaluate on raw data
ari = adjusted_rand_score(y_true_numerical, y_predict)
print("Adjusted Rand Index on Raw data:", ari)
nmi = normalized_mutual_info_score(y_true_numerical, y_predict)
print("Normal Mutual information raw data:", nmi)

# Evaluate on normalized data
kmeans_scale = KMeans(n_clusters=5, n_init=50, random_state=50)
cluster_scale = kmeans_scale.fit_predict(gene_data_scaled)  
y_predict = kmeans_scale.labels_
ari_scale = adjusted_rand_score(y_true_numerical, y_predict)
print("Adjusted Rand Inde on standalized data x:", ari_scale)
nmi_scale = normalized_mutual_info_score(y_true_numerical, y_predict)
print("Normal Mutual information standardized data:", nmi_scale)

#=======Method 2===========
# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data)
# Perform K-means clustering on the PCA results
kmeans_pca = KMeans(n_clusters=5, n_init=10, random_state=50)
clusters_predict = kmeans_pca.fit_predict(X_pca)  

# Evaluate kmeans on pca with Rand Index and Normalized Mutual Information
ari = adjusted_rand_score(y_true_numerical, clusters_predict)  
nmi = normalized_mutual_info_score(y_true_numerical, clusters_predict)
print('Adjusted Rand Index on pca:', ari)
print('Normalized Mutual Information on pca:', nmi)

# Perform t-SNE 
tsne = TSNE(n_components=2, verbose=1, learning_rate=100, n_iter=300, perplexity=100, random_state=50)
x_tsne = tsne.fit_transform(data)
# Perform K-means clustering on the tsne results
kmeans_tsne = KMeans(n_clusters=5, n_init=10, random_state=50)
clusters_predicts_tsne = kmeans_tsne.fit_predict(x_tsne)  

# Evaluate kmeans on tsne with Rand Index and Normalized Mutual Information
ari = adjusted_rand_score(y_true_numerical, clusters_predicts_tsne)  
nmi = normalized_mutual_info_score(y_true_numerical, clusters_predicts_tsne)
print('Adjusted Rand Index on TSNE:', ari)
print('Normalized Mutual Information on TSNE:', nmi)


