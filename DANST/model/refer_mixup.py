import os
import sys
import pandas as pd
import anndata as ad
import numpy as np
import scipy
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')
import random
from model.utils import *
from scipy.sparse import csr_matrix
from scipy.stats import lognorm
from model.data_downsample import downsample_matrix_by_cell

class ReferMixup(object):
    def __init__(self, option_list):
        self.data_path = option_list['data_dir']
        self.ref_dataset_name = option_list['ref_dataset_name']
        self.target_dataset_name = option_list['target_dataset_name']
        self.random_type = option_list['random_type']
        self.type_list = option_list['type_list']
        self.sample_list = option_list['sample_list']
        self.train_sample_num = option_list['ref_sample_num']
        self.sample_size = option_list['sample_size']
        self.HVP_num = option_list['HVP_num']
        self.target_type = option_list['target_type']
        self.target_sample_num = option_list['target_sample_num']
        self.outdir = option_list['SaveResultsDir']
        self.normalize = False #'min-max' 
        self.marker_genes = option_list['DVG_num']
        self.down_sample = option_list['down_sample'] 
   
    def mixup(self):
        if self.sample_list == False:
            # [1] Read and preprocess SC_data 
            sc_data = self.load_ref_dataset(self.ref_dataset_name)
            # [2] Use SC_data to get simulated data
            simulated_data = self.mixup_dataset1(sc_data, self.sample_size, self.train_sample_num)
        else:
            simulated_data = self.read_sm_data()
        # [3] Use SC_data to get simulated data
        if self.target_type == "simulated": 
            target_data_x, target_data_y = self.mixup_dataset(self.target_dataset_name, self.target_sample_num)
            target_data = ad.AnnData(X=target_data_x.to_numpy(), obs=target_data_y)
            target_data.var_names = target_data_x.columns

        elif self.target_type == "real": 
            target_data = self.load_real_data(self.target_dataset_name)
            target_data.var_names_make_unique()
        # [4]Find commom genes in Simulated_data and ST_data
        used_features = self.align_features(simulated_data, target_data)  
        simulated_data = simulated_data[:,used_features]
        target_data = target_data[:,used_features] 
        #[5]Downsample simulated spots
        if self.down_sample:
            train_data = self.downsample_sm_spot_counts(simulated_data,target_data, 6)
        else: train_data = simulated_data
        return train_data, target_data 

    ##1. Read and preprocess SC_data 
    def load_ref_dataset(self, dataset):
        print("1.read SC data")
        filename = os.path.join(self.data_path, dataset)
        try:
            data_h5ad = ad.read_h5ad(filename)
            data_h5ad.var_names_make_unique()
            # Extract celltypes
            if self.type_list == None:
                self.type_list = list(set(data_h5ad.obs[self.random_type].tolist()))
            data_h5ad = data_h5ad[data_h5ad.obs[self.random_type].isin(self.type_list)]
        except FileNotFoundError as e:
            print(f"No such h5ad file found for [cyan]{dataset}")
            sys.exit(e)
        print("1.1 preprocess SC data")
        data_h5ad=preprocess_SC(data_h5ad,n_top_genes= self.HVP_num,celltype=self.random_type )
        genelists = data_h5ad.uns['rank_genes_groups']['names']
        df_genelists = pd.DataFrame.from_records(genelists)
        num_markers = self.marker_genes
        res_genes = []
        for column in df_genelists.head(num_markers): 
            res_genes.extend(df_genelists.head(num_markers)[column].tolist())
        res_genes_ = list(set(res_genes))
        print('Selected DVG Feature Gene number for sc data',len(res_genes_))
        data_h5ad = data_h5ad[:,res_genes_]
        data_h5ad = data_h5ad.copy()
        if scipy.sparse.issparse(data_h5ad.X):
            data_h5ad.X = pd.DataFrame(data_h5ad.X.todense()).fillna(0)
        else:
            data_h5ad.X = pd.DataFrame(data_h5ad.X).fillna(0)
        return data_h5ad   

    ##2.1 Use SC_data to get simulated data
    def mixup_dataset1(self, sc_adata, sample_size, sample_num):
        print("2.1 use SC_data to simulate data")
        mat_sc = sc_adata.X 
        lab_sc_sub = sc_adata.obs[self.random_type]
        
        sc_sub_dict = dict(zip(range(len(self.type_list)), self.type_list))
        sc_sub_dict2 = dict((y, x) for x, y in sc_sub_dict.items()) 
        lab_sc_num = [sc_sub_dict2[ii] for ii in lab_sc_sub] 
        lab_sc_num = np.asarray(lab_sc_num, dtype='int')

        sim_data_x, sim_data_y = self.random_mix(mat_sc, lab_sc_num, nmix=sample_size, n_samples=sample_num, seed=0)
        sim_data_x = pd.DataFrame(sim_data_x,columns = sc_adata.var_names.tolist())
        sim_data_y = pd.DataFrame(sim_data_y,columns = self.type_list)
        if self.normalize:  
            sim_data_x_scale = preprocess_sm(sim_data_x, normalize_method=self.normalize)
            sim_data_x_scale = pd.DataFrame(sim_data_x_scale, columns=sim_data_x.columns)
            sim_data_x = sim_data_x_scale   
        sim_data_x.to_csv(self.outdir+'sim_data_x.csv', index=False,columns = sim_data_x.columns)
        sim_data_y.to_csv(self.outdir+'sim_data_y.csv', index=False,columns = self.type_list)
        sim_data = ad.AnnData(
            X=sim_data_x.to_numpy(),
            obs=sim_data_y
        )
        sim_data.uns["cell_types"] = self.type_list
        sim_data.var_names = sc_adata.var_names
        sim_data.write(self.outdir+'sim_adata.h5ad')
        return sim_data
    
    ##2.2 Read simulated data directly(if already have simulated data)
    def read_sm_data(self):
        print("2.2 directly read simulated_data")
        sim_data=sc.read(self.outdir+'sim_adata.h5ad')
        return sim_data 
    
    ##3.Read and preprocess ST_data
    def load_real_data(self, dataset):
        print("3.read ST data")
        filename = os.path.join(self.data_path, dataset)
        try:
            data_h5ad = ad.read_h5ad(filename)
            data_h5ad.var_names_make_unique()
        except FileNotFoundError as e:
            print(f"No such h5ad file found for [cyan]{dataset}.")
            sys.exit(e)
        print("3.1 preprocess ST data")
        data_h5ad=preprocess_ST(data_h5ad, n_top_genes=self.HVP_num)
        if scipy.sparse.issparse(data_h5ad.X):
            data_h5ad.X = pd.DataFrame(data_h5ad.X.todense()).fillna(0)
        else:
            data_h5ad.X = pd.DataFrame(data_h5ad.X).fillna(0)
        return data_h5ad
    
    ##4.Find commom genes in Simulated_data and ST_data
    def align_features(self, train_data, target_data):
        print("4、find commom genes in Simulated_data and ST_data")
        used_features = list(set(train_data.var_names.tolist()).intersection(
            set(target_data.var_names.tolist())))  # overlapped features between reference and target 
        print("the lenth of selected genes for Real ST and Simulated ST data is",len(used_features))
        return used_features  
    
    ##5.Downsample simulated_data
    def downsample_sm_spot_counts(self, sm_ad,st_ad, n_threads):
        print("5.downsample simulated_data")
        lib_sizes = np.array(st_ad.X.sum(1)).reshape(-1)
        params = lognorm.fit(lib_sizes, floc=0)  # Fit log-normal distribution
        loc = np.log(params[2])  # Estimate location parameter from shape and scale
        scale = params[0]  # Use estimated scale parameter

        sm_mtx_count = sm_ad.X
        sample_cell_counts = np.random.lognormal(loc,scale,sm_ad.shape[0])
        sm_mtx_count_lb = downsample_matrix_by_cell(sm_mtx_count,sample_cell_counts.astype(np.int64), n_cpus=n_threads, numba_end=False)
        sm_ad.X = sm_mtx_count_lb
        return sm_ad
    
    def random_mix(self, Xs, ys, nmix, n_samples, seed):
        # Define empty lists
        Xs_new, ys_new = [], []
        ys = torch.from_numpy(ys)
        ys_ = torch.nn.functional.one_hot(ys)
        ys_ = ys_.numpy()
        np.random.seed(seed)
        random.seed(seed)
        rstate = np.random.RandomState(seed)
        fraction_all = rstate.rand(n_samples, nmix)
        randindex_all = rstate.randint(Xs.shape[0], size=(n_samples, nmix)) 

        for i in range(n_samples):
            # ①fraction: random fraction across the "nmix" number of sampled cells
            fraction = fraction_all[i] 
            fraction = fraction / np.sum(fraction) 
            fraction = np.reshape(fraction, (nmix, 1))  

            # ②Random selection of the single cell data by the index
            randindex = randindex_all[i]  
            ymix = ys_[randindex]  
            # ③Calculate the fraction of cell types in the cell mixture
            yy = np.sum(ymix * fraction, axis=0) 
            # ④Calculate weighted gene expression of the cell mixture
            XX = np.asarray(Xs[randindex]) * fraction 
            XX_ = np.sum(XX, axis=0) 
            # ⑤Add cell type fraction & composite gene expression in the list
            ys_new.append(yy) 
            Xs_new.append(XX_)
        
        Xs_new = np.asarray(Xs_new)
        print(Xs_new.shape)
        ys_new = np.asarray(ys_new)

        return Xs_new, ys_new
    
    def normal_distribution_values(self, mean, std_dev, n_samples, seed):
        np.random.seed(seed)
        random_numbers = np.random.normal(mean, std_dev, size=n_samples) 
        random_integers = np.round(random_numbers).astype(int) 
        random_integers = [num for num in random_integers if num > 0]

        while len(random_integers) < n_samples:
            new_num = np.round(np.random.normal(mean, std_dev)).astype(int)
            if new_num > 0:
                random_integers.append(new_num)

        return random_integers
