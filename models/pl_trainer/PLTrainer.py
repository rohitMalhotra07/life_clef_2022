import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from models.mlp import MLP
from models.ResNetBased import ResnetBasedModel

from dataloader.EnvVectorDataloader import EnvVectorData
from dataloader.ImageDataLoader import MultiChannelImgData
import torch.nn as nn
import numpy as np
from GLC.metrics import predict_top_30_set
from GLC.metrics import top_k_error_rate_from_sets
from typing import List



class EnvVectorModelTrainer(pl.LightningModule):
    def __init__(self,
                 dropout_mlp: float,
                 input_dim: int,
                 n_classes: int,
                 hidden_dims: List[int],
                 l_rate: float,
                 batch_size:int,
                 X_train,
                 y_train,
                 X_val,
                 y_val,
                 num_workers,
                 ):
        super().__init__()
        
        self.model = MLP(dropout_mlp,
                 input_dim,
                 n_classes,
                 hidden_dims)
        
        self.l_rate = l_rate
        self.batch_size = batch_size
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.n_samples_train = self.X_train.shape[0]
        self.n_samples_vals = self.X_val.shape[0]
        self.n_classes = n_classes
        
        # self.train_preds = np.empty((self.n_samples_train, 30))
        self.val_preds = np.empty((self.n_samples_vals, 30))
        
        self.criterion = nn.NLLLoss()
        self.num_workers = num_workers

    def forward(self, x_cont):
        return self.model(x_cont)

    def train_dataloader(self):
        train_dataset = EnvVectorData(self.X_train, self.y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_dataset = EnvVectorData(self.X_val, self.y_val)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size,num_workers=self.num_workers)
        return val_loader

    def training_step(self, batch, batch_idx):
        x_cont, y = batch
        outputs = self.forward(x_cont)
        # print(outputs,x_cont)
        
        # print(y.min(), y.max())
        loss = self.criterion(outputs, y)
        # print(loss)
        
        # s_pred = predict_top_30_set(torch.exp(outputs).cpu())
        # self.train_preds[batch_idx:batch_idx+self.batch_size] = s_pred
        
        self.log("train/step/loss", loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        pass
#         score_val = top_k_error_rate_from_sets(self.y_train, self.train_preds)
#         print("Top-30 error rate: {:.1%} Train Set".format(score_val))
        
#         self.train_preds = np.empty((self.n_samples_train, self.n_classes))
#         self.log("train/epoch/top_30_error", score_val)

    def validation_step(self, batch, batch_idx):
        x_cont, y = batch
        outputs = self.forward(x_cont)
        
        # print(y.min(), y.max())
        val_loss = self.criterion(outputs, y)
        
        # print(outputs.shape)
        # torch.exp(outputs)
        s_pred = predict_top_30_set(outputs.cpu())
        # print(s_pred)
        l = s_pred.shape[0]
        self.val_preds[batch_idx:batch_idx+l] = s_pred
        
        self.log("valid/step/val_loss", val_loss)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        # print(self.val_preds.dtype)
        # print(self.val_preds.astype(int))
        score_val = top_k_error_rate_from_sets(self.y_val, self.val_preds.astype(int))
        print("Top-30 error rate: {:.1%} Validation Set".format(score_val))
        
        self.log("valid/epoch/top_30_error", score_val)
        self.val_preds = np.empty((self.n_samples_vals, 30))
        return {'top_30_error': score_val}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.l_rate)
    

    
    
class ResNetBasedModelTrainer(pl.LightningModule):
    def __init__(self,
                 res_net_out_dim:int,
                 dropout_mlp: float,
                 input_dim: int,
                 n_classes: int,
                 hidden_dims: List[int],
                 l_rate: float,
                 batch_size:int,
                 train_df,
                 val_df,
                 env_vec_columns, 
                 landcover_mapping, 
                 data_path,
                 num_workers,
                 ):
        super().__init__()
        
        self.model = ResnetBasedModel(
                 res_net_out_dim,
                 dropout_mlp,
                 input_dim,
                 n_classes,
                 hidden_dims)
        
        self.l_rate = l_rate
        self.batch_size = batch_size
        
        self.train_df = train_df
        self.val_df = val_df
        
        self.y_val = self.val_df["species_id"].values
        
        self.env_vec_columns = env_vec_columns
        self.landcover_mapping = landcover_mapping
        self.data_path = data_path
        
        self.n_samples_train = self.train_df.shape[0]
        self.n_samples_vals = self.val_df.shape[0]
        self.n_classes = n_classes
        
        # self.train_preds = np.empty((self.n_samples_train, 30))
        self.val_preds = np.empty((self.n_samples_vals, 30))
        
        self.criterion = nn.NLLLoss()
        self.num_workers = num_workers

    def forward(self, img_data, x_cont):
        return self.model(img_data, x_cont)

    def train_dataloader(self):
        train_dataset = MultiChannelImgData(self.train_df, self.env_vec_columns, self.landcover_mapping, self.data_path)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_dataset = MultiChannelImgData(self.val_df, self.env_vec_columns, self.landcover_mapping, self.data_path)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size,num_workers=self.num_workers,shuffle=False)
        return val_loader

    def training_step(self, batch, batch_idx):
        img_data, x_cont, y = batch
        outputs = self.forward(img_data, x_cont)
        # print(outputs,x_cont)
        
        # print(y.min(), y.max())
        loss = self.criterion(outputs, y)
        # print(loss)
        
        # s_pred = predict_top_30_set(torch.exp(outputs).cpu())
        # self.train_preds[batch_idx:batch_idx+self.batch_size] = s_pred
        
        self.log("train/step/loss", loss)
        # return {'loss': loss}
        return loss

    def training_epoch_end(self, outputs):
        pass
#         score_val = top_k_error_rate_from_sets(self.y_train, self.train_preds)
#         print("Top-30 error rate: {:.1%} Train Set".format(score_val))
        
#         self.train_preds = np.empty((self.n_samples_train, self.n_classes))
#         self.log("train/epoch/top_30_error", score_val)

    def validation_step(self, batch, batch_idx):
        img_data, x_cont, y = batch
        outputs = self.forward(img_data, x_cont)
        
        # print(y.min(), y.max())
        val_loss = self.criterion(outputs, y)
        
        # print(outputs.shape)
        # torch.exp(outputs)
        s_pred = predict_top_30_set(outputs.cpu())
        
        l = s_pred.shape[0]
        # print([(batch_idx*self.batch_size),(batch_idx*self.batch_size)+l])
        # print(outputs, torch.exp(outputs))
        self.val_preds[(batch_idx*self.batch_size):(batch_idx*self.batch_size)+l] = s_pred
        
        self.log("valid/step/val_loss", val_loss)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        # print(self.val_preds.dtype)
        # print(self.val_preds)
        score_val = top_k_error_rate_from_sets(self.y_val, self.val_preds.astype(int))
        print("Top-30 error rate: {:.1%} Validation Set".format(score_val))
        
        self.log("valid/epoch/top_30_error", score_val)
        self.val_preds = np.empty((self.n_samples_vals, 30))
        return {'top_30_error': score_val}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.l_rate)