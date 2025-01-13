import os
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import lightning as L
import torchmetrics

class MultiOutputModel(nn.Module):
    def __init__(self, input_shape, hidden_size,
                 intermediate_layer_1, intermediate_layer_2,
                 output_shape_1, output_shape_2,
                 dropout_rate):
        super(MultiOutputModel, self).__init__()
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
        self.intermediate_1 = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(512 if i == 0 else 256, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(dropout_rate)
                ) for i in range(intermediate_layer_1)
            ]
        )

        self.intermediate_2 = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(512 if i == 0 else 256, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(dropout_rate)
                ) for i in range(intermediate_layer_2)
            ]
        )

        self.output_1 = nn.Linear(256, output_shape_1)
        self.output_2 = nn.Linear(256, output_shape_2)
        # self.sigm = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
                     
    def forward(self, x):
        x = self.norm(x)
        x = self.hidden_layers(x)
        x_1 = self.intermediate_1(x)
        x_2 = self.intermediate_2(x)
        out_1 = self.output_1(x_1)
        out_2 = self.output_2(x_2)
        return out_1, out_2

class LitMultiOuputModel(L.LightningModule):
    def __init__(self, model, num_desc_classes, num_grade):
        super().__init__()
        self.model = model
        ### for description
        self.train_f1_score_macro_desc = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_desc_classes, average='macro')        
        self.val_f1_score_macro_desc = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_desc_classes, average='macro')
        self.train_f1_score_micro_desc = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_desc_classes, average='micro')
        self.val_f1_score_micro_desc = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_desc_classes, average='micro')
       
        ## for severity      
        self.train_f1_score_macro_grade = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_grade, average='macro')
        self.val_f1_score_macro_grade = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_grade, average='macro')
        self.train_f1_score_micro_grade = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_grade, average='micro')
        self.val_f1_score_micro_grade = torchmetrics.classification.MulticlassF1Score(
            num_classes=num_grade, average='micro')

    def training_step(self, batch, batch_idx):
        X_drug, y_severity, y_desc = batch
        y_desc = y_desc.argmax(dim=1)
        y_severity = y_severity.argmax(dim=1)
        pred_severity, pred_desc = self.model(X_drug)
        loss_severity = nn.functional.cross_entropy(pred_severity, y_severity)
        loss_desc = nn.functional.cross_entropy(pred_desc, y_desc)
        
        loss = (0.05 * loss_severity) + (0.95 * loss_desc)
        self.log('train_loss', loss, prog_bar=True)
        self.train_f1_score_macro_desc(pred_desc, y_desc)
        self.train_f1_score_micro_desc(pred_desc, y_desc)
        ##
        self.train_f1_score_macro_grade(pred_severity, y_severity)
        self.train_f1_score_micro_grade(pred_severity, y_severity)
        ##
        
        self.log('train_f1_score_macro_desc',self.train_f1_score_macro_desc, prog_bar=True, 
                 on_step=False, on_epoch=True)
        self.log('train_f1_score_micro_desc',self.train_f1_score_micro_desc, prog_bar=True, 
                 on_step=False, on_epoch=True)
        ##
        self.log('train_f1_score_macro_grade',self.train_f1_score_macro_grade, prog_bar=True, 
                 on_step=False, on_epoch=True)
        self.log('train_f1_score_micro_grade',self.train_f1_score_micro_grade, prog_bar=True, 
                 on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        X_drug, y_severity, y_desc = batch
        y_desc = y_desc.argmax(dim=1)
        y_severity = y_severity.argmax(dim=1)
        pred_severity, pred_desc = self.model(X_drug)
        loss_severity = nn.functional.cross_entropy(pred_severity, y_severity)
        loss_desc = nn.functional.cross_entropy(pred_desc, y_desc)
        
        loss = (0.05 * loss_severity) + (0.95 * loss_desc)
        self.log('validation_loss', loss, prog_bar=True)
        self.val_f1_score_macro_desc(pred_desc, y_desc)
        self.val_f1_score_micro_desc(pred_desc, y_desc)
        ##
        self.val_f1_score_macro_grade(pred_severity, y_severity)
        self.val_f1_score_micro_grade(pred_severity, y_severity)
        ##
        
        ##
        self.log('val_f1_score_macro_desc',self.val_f1_score_macro_desc, prog_bar=True, 
                    on_step=False, on_epoch=True)
        self.log('val_f1_score_micro_desc',self.val_f1_score_micro_desc, prog_bar=True, 
                    on_step=False, on_epoch=True)
        ##
        self.log('val_f1_score_macro_grade',self.val_f1_score_macro_grade, prog_bar=True, 
                 on_step=False, on_epoch=True)
        self.log('val_f1_score_micro_grade',self.val_f1_score_micro_grade, prog_bar=True, 
                 on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X_drug, _, _, _ = batch  # Unpacking based on expected input
        pred_severity, pred_desc = self.model(X_drug)
        return pred_severity, pred_desc
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer