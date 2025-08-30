import os
from sklearn import preprocessing as pp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
import scanpy as sc
import torch
import torch.nn as nn
from torch.backends import cudnn
import random
import torch.nn.functional as F
from sklearn.decomposition import PCA

### 1.loss function ###
def L1_loss(preds, gt):
    loss = torch.mean(torch.reshape(torch.square(preds - gt), (-1,)))
    return loss

def KLD_loss(preds, gt):
    preds = F.softmax(preds, dim=1)  
    gt = F.softmax(gt, dim=1) 
    loss = F.kl_div(preds.log(), gt, reduction='mean')
    return loss

def Recon_loss(recon_data, input_data):
    loss_rec_fn = nn.MSELoss().cuda()
    loss= loss_rec_fn(recon_data, input_data)
    return loss

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)

    kernel = torch.exp(-torch.square(tiled_x - tiled_y).mean(dim=2) / dim)
    return kernel

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)

    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd

def kl_divergence(y_true, y_pred, dim=0):
    y_pred = torch.clip(y_pred, torch.finfo(torch.float32).eps)
    y_true = y_true.to(y_pred.dtype)
    y_true = torch.nan_to_num(torch.div(y_true, y_true.sum(dim, keepdims=True)),0)
    y_pred = torch.nan_to_num(torch.div(y_pred, y_pred.sum(dim, keepdims=True)),0)
    y_true = torch.clip(y_true, torch.finfo(torch.float32).eps, 1)
    y_pred = torch.clip(y_pred, torch.finfo(torch.float32).eps, 1)
    return torch.mul(y_true, torch.log(torch.nan_to_num(torch.div(y_true, y_pred)))).mean(dim)

### 2.evaluate metrics ###
def ccc(preds, gt):
    numerator = 2 * np.corrcoef(gt, preds)[0][1] * np.std(gt) * np.std(preds)
    denominator = np.var(gt) + np.var(preds) + (np.mean(gt) - np.mean(preds)) ** 2
    ccc_value = numerator / denominator
    return ccc_value

def compute_metrics(preds, gt):
    gt = gt[preds.columns] # Align pred order and gt order  
    x = pd.melt(preds)['value']
    y = pd.melt(gt)['value']
    CCC = ccc(x, y)
    RMSE = sqrt(mean_squared_error(x, y))
    Corr = pearsonr(x, y)[0]
    return CCC, RMSE, Corr

### 3.preprocess ###
def preprocess_ST(adata,n_top_genes=3000):
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata) 
    # sc.pp.scale(adata,  zero_center=False, max_value=1.0)
    return adata

def preprocess_SC(adata,n_top_genes=3000,celltype=None):
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)  
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)  
    sc.tl.rank_genes_groups(adata, celltype, method='wilcoxon')
    return adata
    

def preprocess_sm(data, normalize_method):
    data = pp.normalize(data, norm='l1') 
    data = np.log1p(data)
    mm = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
    if normalize_method == 'min_max':
        # it scales features so transpose is needed
        data = mm.fit_transform(data.T).T
    elif normalize_method == 'z_score':
        # Z score normalization
        data = (data - data.mean(0))/(data.std(0)+(1e-10))
    return data

### 4.save plot ###
def SaveLossPlot(SavePath, metric_logger, loss_type, output_prex):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    for i in range(len(loss_type)):
        plt.subplot(2, 3, i+1)
        plt.plot(metric_logger[loss_type[i]])
        plt.title(loss_type[i], x = 0.5, y = 0.5)
    imgName = os.path.join(SavePath, output_prex +'.png')
    plt.savefig(imgName)
    plt.close()

def SavePredPlot(SavePath, target_preds, ground_truth):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    celltypes = list(target_preds.columns)

    plt.figure(figsize=(5*(len(celltypes)+1), 5)) 

    eval_metric = []
    x = pd.melt(target_preds)['value']
    y = pd.melt(ground_truth)['value']
    eval_metric.append(ccc(x, y))
    eval_metric.append(sqrt(mean_squared_error(x, y)))
    eval_metric.append(pearsonr(x, y)[0])
    plt.subplot(1, len(celltypes)+1, 1)
    plt.xlim(0, max(y))
    plt.ylim(0, max(y))
    plt.scatter(x, y, s=2)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x,p(x),"r--")
    text = f"$CCC = {eval_metric[0]:0.3f}$\n$RMSE = {eval_metric[1]:0.3f}$\n$Corr = {eval_metric[2]:0.3f}$"
    plt.text(0.05, max(y)-0.05, text, fontsize=8, verticalalignment='top')
    plt.title('All samples')
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')

    for i in range(len(celltypes)):
        eval_metric = []
        x = target_preds[celltypes[i]]
        y = ground_truth[celltypes[i]]
        eval_metric.append(ccc(x, y))
        eval_metric.append(sqrt(mean_squared_error(x, y)))
        eval_metric.append(pearsonr(x, y)[0])
        plt.subplot(1, len(celltypes)+1, i+2)
        plt.xlim(0, max(y))
        plt.ylim(0, max(y))
        plt.scatter(x, y, s=2)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x,p(x),"r--")
        text = f"$CCC = {eval_metric[0]:0.3f}$\n$RMSE = {eval_metric[1]:0.3f}$\n$Corr = {eval_metric[2]:0.3f}$"
        plt.text(0.05, max(y)-0.05, text, fontsize=8, verticalalignment='top')
        plt.title(celltypes[i])
        plt.xlabel('Prediction')
        plt.ylabel('Ground Truth')
    imgName = os.path.join(SavePath, 'pred_fraction_target_scatter.jpg')
    plt.savefig(imgName)
    plt.close()

def SavetSNEPlot(SavePath, ann_data, output_prex):
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    sc.tl.pca(ann_data, svd_solver='arpack')
    sc.pp.neighbors(ann_data, n_neighbors=10, n_pcs=20)
    sc.tl.tsne(ann_data)
    with plt.rc_context({'figure.figsize': (8, 8)}):
        sc.pl.tsne(ann_data, color=list(ann_data.obs.columns), color_map='viridis',frameon=False)
        plt.tight_layout()
    plt.savefig(os.path.join(SavePath, output_prex + "_tSNE_plot.jpg"))
    ann_data.write(os.path.join(SavePath, output_prex + ".h5ad"))
    plt.close()

### 5.set random seed ###
def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 

### 6、clustering ###
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=1400):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False,random_seed=1400):
    pca = PCA(n_components=25, random_state=42)
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding
    
    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters,random_seed=random_seed)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain'] 
       
    if refinement:  
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['domain'] = new_type 
def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
   
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res    