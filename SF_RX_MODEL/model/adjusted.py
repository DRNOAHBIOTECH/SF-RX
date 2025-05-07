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
                    nn.Linear(input_shape*2 if i == 0 else 512, 512),
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
        self.output_3 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(dropout_rate),
                    nn.Linear(256, 1))
        self.sigm = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)
                     
    def forward(self, drug_a_feat, drug_b_feat):
        # Normalize inputs
        drug_a_feat = self.norm(drug_a_feat)
        drug_b_feat = self.norm(drug_b_feat)

        # Concatenate features of both drugs for both directions (A -> B and B -> A)
        a_to_b_combined = torch.cat([drug_a_feat, drug_b_feat], dim=1)
        b_to_a_combined = torch.cat([drug_b_feat, drug_a_feat], dim=1)
        
        # Shared hidden layers (same for both drugs)
        a_to_b_combined = self.hidden_layers(a_to_b_combined)
        b_to_a_combined = self.hidden_layers(b_to_a_combined)
        
        # Process severity prediction for both directions (A -> B and B -> A)
        a_to_b_severity = self.intermediate_1(a_to_b_combined)
        b_to_a_severity = self.intermediate_1(b_to_a_combined)
        # severity_mean = self.softmax((self.output_1(a_to_b_severity) + self.output_1(b_to_a_severity)) / 2)
        severity_mean = (self.output_1(a_to_b_severity) + self.output_1(b_to_a_severity)) / 2
       
        # Process description prediction for both directions (A -> B and B -> A)
        a_to_b_desc = self.intermediate_2(a_to_b_combined)
        b_to_a_desc = self.intermediate_2(b_to_a_combined)
        # desc_mean = self.softmax((self.output_2(a_to_b_desc) + self.output_2(b_to_a_desc)) / 2)
        desc_mean = (self.output_2(a_to_b_desc) + self.output_2(b_to_a_desc)) / 2
        
        # Direction prediction: Concatenate both drug features for binary classification
        direc = self.output_3(a_to_b_combined)
        direc_probs = self.sigm(direc)
       
        return severity_mean, desc_mean, direc_probs

# %%
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
        
        # for direction
        self.train_f1_score_direc = torchmetrics.classification.BinaryF1Score()
        self.val_f1_score_direc = torchmetrics.classification.BinaryF1Score()

    def training_step(self, batch, batch_idx):
        X_drugA, X_drugB, y_severity, y_desc, y_direc = batch
        y_desc = torch.argmax(y_desc, dim=1)
        y_severity = torch.argmax(y_severity, dim=1)
        
        pred_severity, pred_desc, pred_direc = self.model(X_drugA, X_drugB)
        loss_severity = nn.functional.cross_entropy(pred_severity, y_severity)
        loss_desc = nn.functional.cross_entropy(pred_desc, y_desc)
        loss_direc = nn.functional.binary_cross_entropy(pred_direc, y_direc)
        loss = (0.1 * loss_severity) + (0.8 * loss_desc) + (0.1 * loss_direc)
        self.log('train_loss', loss, prog_bar=True)
        self.train_f1_score_macro_desc(pred_desc, y_desc)
        self.train_f1_score_micro_desc(pred_desc, y_desc)
        ##
        self.train_f1_score_macro_grade(pred_severity, y_severity)
        self.train_f1_score_micro_grade(pred_severity, y_severity)
        ##
        self.train_f1_score_direc(pred_direc, y_direc)
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
        ##
        self.log('train_f1_score_direc',self.train_f1_score_direc, prog_bar=True, 
                 on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        X_drugA, X_drugB, y_severity, y_desc, y_direc = batch
        y_desc = torch.argmax(y_desc, dim=1)
        y_severity = torch.argmax(y_severity, dim=1)
        
        pred_severity, pred_desc, pred_direc = self.model(X_drugA, X_drugB)
        loss_severity = nn.functional.cross_entropy(pred_severity, y_severity)
        loss_desc = nn.functional.cross_entropy(pred_desc, y_desc)
        loss_direc = nn.functional.binary_cross_entropy(pred_direc, y_direc)
        loss = (0.1 * loss_severity) + (0.8 * loss_desc) + (0.1 * loss_direc)
        self.log('validation_loss', loss, prog_bar=True)
        self.val_f1_score_macro_desc(pred_desc, y_desc)
        self.val_f1_score_micro_desc(pred_desc, y_desc)
        ##
        self.val_f1_score_macro_grade(pred_severity, y_severity)
        self.val_f1_score_micro_grade(pred_severity, y_severity)
        ##
        self.val_f1_score_direc(pred_direc, y_direc)
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
        ##
        self.log('val_f1_score_direc',self.val_f1_score_direc, prog_bar=True, 
                 on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X_drugA, X_drugB, _, _, _ = batch  # Unpacking based on expected input
        pred_severity, pred_desc, pred_direc = self.model(X_drugA, X_drugB)
        return pred_severity, pred_desc, pred_direc
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer