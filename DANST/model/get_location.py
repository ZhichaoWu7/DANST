from model.utils import clustering
import scanpy as sc
import numpy as np
from scipy.spatial.distance import cdist

def location(option_list,source_data,target_data,tool,radius=0,random_seed=0,refinement=False):
    n_clusters = option_list['get_location']['n_clusters']
    start =  option_list['get_location']['start']
    end = option_list['get_location']['end']
    increment = option_list['get_location']['increment']
    target_data.obsm['emb']=target_data.X
    if tool == 'mclust':
         clustering(target_data, n_clusters, radius=radius, method=tool, refinement=refinement, random_seed=random_seed) 
    elif tool in ['leiden', 'louvain']:
         clustering(target_data, n_clusters, radius=radius, method=tool, start=start, end=end, increment=increment, refinement=refinement)     
  
    spatial_coords = target_data.obsm['spatial']
    cluster_labels = target_data.obs['domain']

    unique_labels = np.unique(cluster_labels) 
    cluster_centers = np.zeros((len(unique_labels), spatial_coords.shape[1]))
    cluster_features = np.zeros((len(unique_labels), target_data.n_vars))
    for i, label in enumerate(unique_labels):
        cluster_mask = cluster_labels == label
        cluster_centers[i] = np.mean(spatial_coords[cluster_mask], axis=0)
        cluster_features[i] = np.mean(target_data.X[cluster_mask], axis=0)
    
    sc_features = source_data.X
    feature_distances = cdist(sc_features, cluster_features)

    weights = 1 / (feature_distances + 1e-6)
    source_data.obsm['spatial'] = np.dot(weights, cluster_centers) / np.sum(weights, axis=1, keepdims=True)
    source_data.write(option_list['SaveResultsDir']+'located_sim_adata.h5ad')
    return source_data
