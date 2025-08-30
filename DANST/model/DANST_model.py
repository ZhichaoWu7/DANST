import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as Data
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model.utils import *
from model.lr_scheduler import *
import torch.nn.init as init
from model.Block import *


class DANST(object):
    def __init__(self, option_list):
        self.num_epochs = option_list['DANN_epochs']
        self.batch_size = option_list['batch_size']
        self.target_type = option_list['target_type']
        self.learning_rate1 = option_list['DANN_learning_rate1']
        self.learning_rate2 = option_list['DANN_learning_rate2']
        self.celltype_num = None
        self.labels = None
        self.used_features = None
        self.seed = 42
        self.outdir = option_list['SaveResultsDir']
        self.weight_decay = 3e-4

        random_seed(self.seed)

    def DANST_model(self, celltype_num, trained_predictor):
        feature_num = len(self.used_features)
        
        self.encoder_da = nn.Sequential(FCNBlock(feature_num, 256),
                                        nn.Linear(256, 128)) #Encoder2

        # self.predictor_da = nn.Sequential(FCNBlock(128, 64),
        #                                   nn.Linear(64, celltype_num), 
        #                                   nn.Softmax(dim=1))#Predictor2
        self.predictor_da = trained_predictor
        
        self.discriminator_da = nn.Sequential(FCNBlock(128, 64),
                                              nn.Linear(64, 1),
                                              nn.Sigmoid()) #Domain discrininator
        nn.init.xavier_uniform_(self.discriminator_da[1].weight)  

        model_da = nn.ModuleList([])
        model_da.append(self.encoder_da)
        model_da.append(self.predictor_da)
        model_da.append(self.discriminator_da)
        return model_da 
    
    ### Prepare data loader for training ###
    def prepare_dataloader(self, source_data, target_data, batch_size):    
        source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
        self.source_data_x = source_data.X.astype(np.float32) 
        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose() 
        tr_data = torch.FloatTensor(self.source_data_x)
        tr_labels = torch.FloatTensor(self.source_data_y)
        source_dataset = Data.TensorDataset(tr_data, tr_labels)
        self.train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=batch_size, shuffle=True)

        # Extract celltype and feature info
        self.labels = source_data.uns['cell_types']
        self.celltype_num = len(self.labels)
        self.used_features = list(source_data.var_names)

        self.target_data_x = target_data.X.astype(np.float32) 
        if self.target_type == "simulated":
            target_ratios = [target_data.obs[ctype] for ctype in self.labels]
            self.target_data_y = np.array(target_ratios, dtype=np.float32).transpose()
        elif self.target_type == "real": 
            self.target_data_y = np.random.rand(target_data.shape[0], self.celltype_num) 

        te_data = torch.FloatTensor(self.target_data_x)
        te_labels = torch.FloatTensor(self.target_data_y)
        target_dataset = Data.TensorDataset(te_data, te_labels)
       
        self.train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size = batch_size, shuffle=False)

    def train(self, source_data, target_data, trained_predictor):
       
        pred_loss_min = np.inf
        disc_loss_min = np.inf
        disc_loss_DA_min = np.inf
        curr_step = 0
        patience = 10

        scheduler = "scheduler_ReduceLROnPlateau"

        ### [1]prepare model structure ###
        self.prepare_dataloader(source_data, target_data, self.batch_size)
        self.model_da = self.DANST_model(self.celltype_num, trained_predictor).cuda()

        ### [2]setup optimizer ###
        optimizer_da1 = torch.optim.Adam([{'params': self.encoder_da.parameters()},
                                          {'params': self.predictor_da.parameters()}],
                                         lr=self.learning_rate1)
        optimizer_da2 = torch.optim.Adam([{'params': self.discriminator_da.parameters()}],
                                         lr=self.learning_rate2)

        criterion_da=nn.BCELoss().cuda()
        source_label = torch.ones(self.batch_size).unsqueeze(1).cuda()   # 定义source domain label为1
        target_label = torch.zeros(self.batch_size).unsqueeze(1).cuda()  # 定义target domain label为0   
        
        ### [3]Stage3-train ###
        for epoch in range(self.num_epochs):
            self.model_da.train() 
            train_target_iterator = iter(self.train_target_loader) 
            pred_loss_epoch, disc_loss_epoch, disc_loss_DA_epoch = 0., 0., 0.
            for batch_idx, (source_x, source_y) in enumerate(self.train_source_loader):
                # get batch item of target
                try:
                    target_x, _ = next(train_target_iterator) 
                except StopIteration: 
                    train_target_iterator = iter(self.train_target_loader)
                    target_x, _ = next(train_target_iterator)
                #####——————————————————————[3.1]—————————————————————####
                embedding_source = self.encoder_da(source_x.cuda()) #E2r = Encoder2(Xr')
                embedding_target = self.encoder_da(target_x.cuda()) #E2t = Encoder2(Xt)
                frac_pred = self.predictor_da(embedding_source) #Yr = Predictor2(E2r)
                domain_pred_source = self.discriminator_da(embedding_source) #Dr = discriminator(E2r)
                domain_pred_target = self.discriminator_da(embedding_target) #Dt = discriminator(E2t)

                # Loss_pred 
                pred_loss =  L1_loss(frac_pred, source_y.cuda())+kl_divergence(source_y.cuda(), frac_pred).mean()
                pred_loss_epoch += pred_loss.data.item()
                # LOSS_discriminator
                disc_loss = criterion_da(domain_pred_target,source_label) + criterion_da(domain_pred_source, target_label)
                disc_loss_epoch += disc_loss.data.item()
                loss = pred_loss + disc_loss

                ## update weights
                optimizer_da1.zero_grad()
                loss.backward(retain_graph=True) 
                optimizer_da1.step()
                #####——————————————————————[3.2]—————————————————————####
                if batch_idx % 1 == 0:
                    embedding_source = self.encoder_da(source_x.cuda()) #E2r = Encoder2(Xr')
                    embedding_target = self.encoder_da(target_x.cuda()) #E2t = Encoder2(Xt)
                    domain_pred_source = self.discriminator_da(embedding_source) #Dr = discriminator(E2r)
                    domain_pred_target = self.discriminator_da(embedding_target) #Dt = discriminator(E2t)

                    # LOSS_discriminator
                    disc_loss_DA = criterion_da(domain_pred_target,target_label) + criterion_da(domain_pred_source, source_label)
                    disc_loss_DA_epoch += disc_loss_DA.data.item()
                    loss = disc_loss_DA
                    # update weights
                    optimizer_da2.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer_da2.step()
            pred_loss_epoch = pred_loss_epoch/(batch_idx + 1)
            disc_loss_epoch = disc_loss_epoch/(batch_idx + 1)
            disc_loss_DA_epoch = disc_loss_DA_epoch/(batch_idx + 1)
        
            if (epoch+1) % 20 == 0:
                print('============= Epoch {:02d}/{:02d} in stage3 ============='.format(epoch + 1, self.num_epochs))
                print("pred_loss=%f, disc_loss=%f, disc_loss_DA=%f" % (pred_loss_epoch, disc_loss_epoch, disc_loss_DA_epoch))
                if self.target_type == "simulated": 
                    ### model validation on target data ###
                    target_preds, ground_truth = self.prediction()
                    epoch_ccc, epoch_rmse, epoch_corr = compute_metrics(target_preds, ground_truth)

    ##stage4
    def prediction(self):
        self.model_da.eval() 
        preds, gt = None, None
        for batch_idx, (x, y) in enumerate(self.test_target_loader): 
            logits = self.predictor_da(self.encoder_da(x.cuda())).detach().cpu().numpy()#Yt=Predictor(Encoder(Xt))
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)

        target_preds = pd.DataFrame(preds, columns=self.labels)
        ground_truth = pd.DataFrame(gt, columns=self.labels)
        return target_preds, ground_truth 
    