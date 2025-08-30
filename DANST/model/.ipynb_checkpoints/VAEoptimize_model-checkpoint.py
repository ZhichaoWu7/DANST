import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import random
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model.utils import *
from model.lr_scheduler import *
import torch.nn.init as init
import torch.nn.functional as F
from model.Block import *
      
class VAEoptimize(object):
    def __init__(self, option_list):
        self.num_epochs = option_list['AE_epochs'] 
        self.batch_size = option_list['batch_size']
        self.learning_rate = option_list['AE_learning_rate']
        self.celltype_num = None
        self.labels = None
        self.used_features = None
        self.seed = 42
        self.outdir = option_list['SaveResultsDir']
        random_seed(self.seed)

    def VAEoptimize_model(self, celltype_num):
        feature_num = len(self.used_features)

        self.encoder_im = nn.Sequential(GCNBlock(feature_num, 256), 
                                        VA_GCNBlock(256, 128)) # Encoder1                          
        
        self.predictor_im = nn.Sequential(FCNBlock(128, 64),
                                          nn.Linear(64, celltype_num), 
                                          nn.Softmax(dim=-1)) # Predictor1

        self.decoder_im = nn.Sequential(FCNBlock(128, 256), 
                                        nn.Linear(256,feature_num)) # Decoder1
        
        nn.init.xavier_uniform_(self.predictor_im[1].weight) 
        nn.init.xavier_uniform_(self.decoder_im[1].weight) 
                               
        model_im = nn.ModuleList([])
        model_im.append(self.encoder_im)
        model_im.append(self.predictor_im)
        model_im.append(self.decoder_im)
        return model_im 

    def prepare_dataloader(self, ref_data, target_data,batch_size):
        ### Prepare data loader for training ###
        ref_ratios = [ref_data.obs[ctype] for ctype in ref_data.uns['cell_types']]
        self.ref_data_x = ref_data.X.astype(np.float32) 
        self.ref_data_y = np.array(ref_ratios, dtype=np.float32).transpose() 
        tr_data = torch.FloatTensor(self.ref_data_x)
        tr_labels = torch.FloatTensor(self.ref_data_y)
        ref_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_ref_loader = Data.DataLoader(dataset=ref_dataset, batch_size=batch_size, shuffle=False)
        self.test_ref_loader =  Data.DataLoader(dataset=ref_dataset, batch_size=batch_size, shuffle=False)
        # Extract celltype and feature info
        self.labels = ref_data.uns['cell_types']
        self.celltype_num = len(self.labels)
        self.used_features = list(ref_data.var_names)

        self.target_data_x = target_data.X.astype(np.float32) 
        self.target_data_y = np.random.rand(target_data.shape[0], self.celltype_num)
        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)

        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False,drop_last=False)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=False,drop_last=False)

    def train(self, ref_data, target_data,adj):
 
        scheduler = "scheduler_ReduceLROnPlateau"

        ### [1]prepare model structure ###
        self.prepare_dataloader(ref_data, target_data, self.batch_size)
        self.model_im = self.VAEoptimize_model(self.celltype_num).cuda()

        ### [2]setup optimizer ###
        optimizer_im = torch.optim.Adam([{'params': self.encoder_im.parameters()},
                                         {'params': self.predictor_im.parameters()},
                                         {'params': self.decoder_im.parameters()}],
                                        lr=self.learning_rate)

        scheduler_im = lr_scheduler(scheduler=scheduler, optimizer=optimizer_im)
        metric_logger = defaultdict(list)

        ### [3]Stage2-train ###
        for epoch in range(self.num_epochs):
            self.model_im.train()

            train_target_iterator = iter(self.train_target_loader) 
            loss_epoch, pred_loss_epoch, recon_loss_epoch = 0., 0., 0.
            target_recon, target_label , ref_recon, ref_label = None, None, None, None
            for batch_idx, (ref_x, ref_y) in enumerate(self.train_ref_loader):
                ref_indices = [i + batch_idx * self.batch_size for i in range(ref_x.size(0))]
                # get batch item of target
                try:
                    target_x, target_y = next(train_target_iterator) 
                except StopIteration:
                    train_target_iterator = iter(self.train_target_loader)
                    target_x, target_y = next(train_target_iterator)
                n=int(target_data.n_obs//self.batch_size)+1 
                m =int(batch_idx%n)
                target_sample_indices = [i + m * self.batch_size for i in range(target_x.size(0))]
                target_indices = [idx + ref_data.n_obs for idx in target_sample_indices]
                all_indices = ref_indices + target_indices
                
                adj1 = adj[all_indices, :][:, all_indices].cuda()
                X = torch.cat((ref_x, target_x)).cuda()

                hid_emb = self.encoder_im[0](X,adj1) #[E1r,E1t] = Encoder1[Xr,Xt]
                embedding = self.encoder_im[1](hid_emb,adj1)
                frac_pred = self.predictor_im(embedding) #[Yr,Yt] = Predictor1[E1r,E1t]
                recon_X = self.decoder_im(embedding) #[Xr',Xt'] = Decoder[E1r,E1t]

                # caculate loss 
                pred_loss = L1_loss(frac_pred[range(self.batch_size),], ref_y.cuda())+kl_divergence(ref_y.cuda(), frac_pred[range(self.batch_size),]).mean()
                pred_loss_epoch += pred_loss
                rec_loss = Recon_loss(recon_X, X.cuda()) 
                recon_loss_epoch += rec_loss
                loss = rec_loss + pred_loss 
                loss_epoch += loss   

                # update weights
                optimizer_im.zero_grad()
                loss.backward()
                optimizer_im.step()
                # if scheduler_im != None: 
                #     try:
                #         scheduler_im.step()
                #     except:
                #         scheduler_im.step(metrics=loss)
                if(epoch==self.num_epochs-1):
                    target_recon, target_label , ref_recon, ref_label = self.get_recon(ref_data, target_data,recon_X, ref_y, target_y, target_recon, target_label, ref_recon, ref_label)
                
            loss_epoch = (loss_epoch/(batch_idx + 1))
            metric_logger['cAE_loss'].append(loss_epoch)
            pred_loss_epoch = (pred_loss_epoch/(batch_idx + 1))
            metric_logger['pred_loss'].append(pred_loss_epoch)
            recon_loss_epoch = (recon_loss_epoch/(batch_idx + 1))
            metric_logger['recon_loss'].append(recon_loss_epoch)
            if (epoch+1) % 20 == 0:
                print('============= Epoch {:02d}/{:02d} in stage2 ============='.format(epoch + 1, self.num_epochs))
                print("cAE_loss=%f, pred_loss=%f, recon_loss=%f"% (loss_epoch, pred_loss_epoch, recon_loss_epoch))

        ### [4]Save reconstruction data of ref and target ###
        # ref_recon_data, target_recon_data = self.write_recon(ref_data,adj)
        trained_encoder = self.encoder_im
        trained_predictor = self.predictor_im
        ref_recon_data = sc.read(self.outdir+'ref_recon_data.h5ad')  
        target_recon_data=sc.read(self.outdir+'target_recon_data.h5ad')
        return ref_recon_data, target_recon_data, trained_predictor


    def get_recon(self,ref_data, target_data, recon_X, ref_y,target_y,target_recon, target_label , ref_recon, ref_label):
        if ref_recon is None or (ref_recon is not None and ref_recon.shape[0]<ref_data.n_obs):
            x_recon =  recon_X[:self.batch_size,:].detach().cpu().numpy()
            labels= ref_y.detach().cpu().numpy()
            ref_recon = x_recon if ref_recon is None else np.concatenate((ref_recon, x_recon), axis=0)
            ref_label = labels if ref_label is None else np.concatenate((ref_label, labels), axis=0)
            if  ref_recon.shape[0]==ref_data.n_obs:
                ref_recon = pd.DataFrame(ref_recon, columns=self.used_features) 
                ref_label = pd.DataFrame(ref_label, columns=self.labels) 

                ref_recon_data = ad.AnnData(X=ref_recon.to_numpy(), obs=ref_label)
                ref_recon_data.uns['cell_types'] = self.labels
                ref_recon_data.var_names = self.used_features
                ref_recon_data.write(self.outdir+'ref_recon_data.h5ad')
                
        if target_recon is None or (target_recon is not None and target_recon.shape[0]<target_data.n_obs):
            x_recon_t =  recon_X[self.batch_size:,:].detach().cpu().numpy()
            labels_t = target_y.detach().cpu().numpy()
            target_recon = x_recon_t if target_recon is None else np.concatenate((target_recon, x_recon_t), axis=0)
            target_label = labels_t if target_label is None else np.concatenate((target_label, labels_t), axis=0)
            if target_recon.shape[0]==target_data.n_obs:
                target_recon = pd.DataFrame(target_recon, columns=self.used_features) 
                target_label = pd.DataFrame(target_label, columns=self.labels)
                target_recon_data = ad.AnnData(X=target_recon.to_numpy(), obs=target_label)
                target_recon_data.uns['cell_types'] = self.labels
                target_recon_data.var_names = self.used_features
                target_recon_data.write(self.outdir+'target_recon_data.h5ad')
        
        return target_recon, target_label , ref_recon, ref_label

    def write_recon(self, ref_data,adj):
        
        self.model_im.eval()
        
        ref_recon, ref_label = None, None
        for batch_idx, (ref_x, ref_y) in enumerate(self.test_ref_loader):
            ref_indices = [i + batch_idx * self.batch_size for i in range(ref_x.size(0))]
            ref_adj = adj[ref_indices, :][:, ref_indices].cuda()
            ref_hid_emb = self.encoder_im[0](ref_x.cuda(),ref_adj) 
            ref_x_embedding = self.encoder_im[1](ref_hid_emb,ref_adj)
            ref_x_recon = self.decoder_im(ref_x_embedding).detach().cpu().numpy() 
            labels = ref_y.detach().cpu().numpy()
            ref_recon = ref_x_recon if ref_recon is None else np.concatenate((ref_recon, ref_x_recon), axis=0)
            ref_label = labels if ref_label is None else np.concatenate((ref_label, labels), axis=0)
        ref_recon = pd.DataFrame(ref_recon, columns=self.used_features) 
        ref_label = pd.DataFrame(ref_label, columns=self.labels)

        ref_recon_data = ad.AnnData(X=ref_recon.to_numpy(), obs=ref_label)
        ref_recon_data.uns['cell_types'] = self.labels
        ref_recon_data.var_names = self.used_features
        ref_recon_data.write(self.outdir+'ref_recon_data.h5ad')
        
        target_recon, target_label = None, None
        for batch_idx, (target_x, target_y) in enumerate(self.test_target_loader):
            target_indices = [ref_data.n_obs + i + batch_idx * self.batch_size for i in range(target_x.size(0))]
            target_adj = adj[target_indices, :][:, target_indices].cuda()
            target_hid_emb = self.encoder_im[0](target_x.cuda(),target_adj) 
            target_x_embedding = self.encoder_im[1](target_hid_emb,target_adj)
            target_x_recon = self.decoder_im(target_x_embedding).detach().cpu().numpy()
            labels = target_y.detach().cpu().numpy()
            target_recon = target_x_recon if target_recon is None else np.concatenate((target_recon, target_x_recon), axis=0)
            target_label = labels if target_label is None else np.concatenate((target_label, labels), axis=0)
        target_recon = pd.DataFrame(target_recon, columns=self.used_features)
        target_label = pd.DataFrame(target_label, columns=self.labels)

        target_recon_data = ad.AnnData(X=target_recon.to_numpy(), obs=target_label)
        target_recon_data.uns['cell_types'] = self.labels
        target_recon_data.var_names = self.used_features
        target_recon_data.write(self.outdir+'target_recon_data.h5ad')
        
        return ref_recon_data, target_recon_data 
