import os
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import lightning as L
import torchmetrics

from config import *

class MlpMultiTask(nn.Module): # Baseline model
    def __init__(self, input_shape, hidden_size,
                 intermediate_layer, 
                 output_shape, dropout_rate=0.3):
        super(MlpMultiTask, self).__init__()
        self.norm = nn.BatchNorm1d(input_shape)
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(input_shape if i == 0 else 512, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(dropout_rate)
                ) for i in range(hidden_size)
            ]
        )
        
        self.intermediate = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(512 if i == 0 else 256, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(dropout_rate)
                ) for i in range(intermediate_layer)
            ]
        )
        
        self.output_1 = nn.Linear(256, output_shape)

    def forward(self, x):
        x = self.norm(x)
        x = self.hidden_layers(x)
        x = self.intermediate(x)
        x = self.output_1(x)
        return x

class LitMlpMultiTask(L.LightningModule):
    def __init__(self, model, num_sev, weight_decay):
        super().__init__()
        self.model = model
        self.weight_decay = weight_decay
        
        self.sev_train_f1_macro = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_sev, average='macro')
        self.sev_validation_f1_macro = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_sev, average='macro')
        self.sev_train_f1_micro = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_sev, average='micro')
        self.sev_validation_f1_micro = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_sev, average='micro')
    
    def training_step(self, batch, batch_idx):
        x_batch, y_batch_sev  = batch
        pred_sev = self.model(x_batch)
        loss = nn.CrossEntropyLoss()(pred_sev, y_batch_sev)
        
        y_batch_sev = torch.argmax(y_batch_sev, dim = 1)
        pred_sev = torch.argmax(pred_sev, dim = 1)
        
        self.log('train_loss', loss, prog_bar=True)
        
        self.sev_train_f1_macro(pred_sev, y_batch_sev)
        self.sev_train_f1_micro(pred_sev, y_batch_sev)

        self.log('sev_train_f1_macro',self.sev_train_f1_macro, prog_bar=True, 
                 on_step=False, on_epoch=True)
        self.log('sev_train_f1_micro',self.sev_train_f1_micro, prog_bar=True, 
                 on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch_sev  = batch
        pred_sev = self.model(x_batch)
        loss = nn.CrossEntropyLoss()(pred_sev, y_batch_sev)
        
        y_batch_sev = torch.argmax(y_batch_sev, dim = 1)
        pred_sev = torch.argmax(pred_sev, dim = 1)
        
        self.log('validation_loss', loss, prog_bar=True)

        self.sev_validation_f1_macro(pred_sev, y_batch_sev)
        self.sev_validation_f1_micro(pred_sev, y_batch_sev)

        self.log('sev_validation_f1_macro',self.sev_validation_f1_macro, prog_bar=True, 
                 on_step=False, on_epoch=True)
        self.log('sev_validation_f1_micro',self.sev_validation_f1_micro, prog_bar=True, 
                 on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_batch, _  = batch
        pred_sev = self.model(x_batch)
        return pred_sev
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)
        return optimizer
    
def load_hyperparam(lv, dat_src):
    input_shape = INPUT_SHAPE
    output_shape = OUTOUT_SHAPE[dat_src]
    
    file_path = f'hyperparameter.json'
    with open(file_path, 'r') as f:
        params = json.load(f)

    hyperparams = {
        "input_shape": input_shape,
        "hidden_size": params['hidden_size'],
        "intermediate_layer": params['intermediate_layer_1'],
        "dropout_rate": params['dropout_rate'],
        "output_shape": output_shape
    }

    return hyperparams