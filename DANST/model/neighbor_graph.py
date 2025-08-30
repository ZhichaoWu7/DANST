import ot
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors 
import torch

def construct_interaction(position, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""

    # calculate distance matrix spot
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot]) 
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1 
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj) 
    return adj
    
def construct_interaction_KNN(position, n_neighbors=3): 
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    return adj 

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def calculate_all_adj(option_list, datatype, source_data, target_data):
    n_neighbors = option_list['n_neighbors']
    source_spatial = torch.from_numpy(source_data.obsm['spatial'])
    target_spatial = torch.from_numpy(target_data.obsm['spatial'])
    location = torch.cat([source_spatial, target_spatial], dim=0)
    if datatype in ['Stereo', 'Slide']:
        adj = construct_interaction_KNN(location,n_neighbors)
        adj = preprocess_adj_sparse(adj)
    else:   
        adj=construct_interaction(location,n_neighbors)
        adj = preprocess_adj(adj) 
        adj = torch.FloatTensor(adj)
    return adj

def calculate_each_adj(datatype, adata,n_neighbors=3):
    location = adata.obsm['spatial']
    if datatype in ['Stereo', 'Slide']:
        adj = construct_interaction_KNN(location)
        adj = preprocess_adj_sparse(adj)
    else:   
        adj=construct_interaction(location)
        adj = preprocess_adj(adj)
        adj = torch.FloatTensor(adj)
    adata.obsm['adj']=adj
    return adata


                