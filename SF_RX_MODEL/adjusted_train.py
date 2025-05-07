import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import lightning as L
import torchmetrics

from config import *
from utils import *
sys.path.append("model/")
from adjusted import MultiOutputModel, LitMultiOuputModel

torch.set_float32_matmul_precision('medium')

def main(lv, device): 
    X_path = 'data/toy_conc_X.csv'
    y_path = 'data/toy_conc_y.csv'

    X, y = load_data(X_path, y_path)
    
    for val_fold in range(5):    
        val_csv_logger = CSVLogger(save_dir=RESULT_PATH + f'lv{lv}/Adjusted/val_fold{val_fold}')
        
        if lv != 2:
            val_X, val_y, tr_X, tr_y = split_data(X, y, val_fold, lv)
        else:
            val_X, val_y, rows_to_remove = adj_lv2_split_validation_data(X, y, val_fold)
            tr_X, tr_y = adj_lv2_split_training_data(X, y, val_fold, rows_to_remove)
        
        val_druga_X = torch.tensor(val_X.iloc[:,:509].values, dtype=torch.float32)
        val_drugb_X = torch.tensor(val_X.iloc[:,509:].values, dtype=torch.float32)

        tr_druga_X = torch.tensor(tr_X.iloc[:,:509].values, dtype=torch.float32)
        tr_drugb_X = torch.tensor(tr_X.iloc[:,509:].values, dtype=torch.float32)

        y_val_severity = torch.tensor(pd.get_dummies(val_y.severity).values, dtype=torch.float32)
        y_val_desc = torch.tensor(pd.get_dummies(val_y.type).values, dtype=torch.float32)
        y_val_direc = torch.tensor(val_y.direction.values, dtype=torch.float32).unsqueeze(1)

        y_tr_severity = torch.tensor(pd.get_dummies(tr_y.severity).values, dtype=torch.float32)
        y_tr_desc = torch.tensor(pd.get_dummies(tr_y.type).values, dtype=torch.float32)
        y_tr_direc = torch.tensor(tr_y.direction.values, dtype=torch.float32).unsqueeze(1)

        val_dataset = TensorDataset(val_druga_X, val_drugb_X,
                                    y_val_severity, y_val_desc, y_val_direc)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0, 
                                pin_memory=torch.cuda.is_available())

        tr_dataset = TensorDataset(tr_druga_X, tr_drugb_X,
                                    y_tr_severity, y_tr_desc, y_tr_direc)
        tr_loader = DataLoader(tr_dataset, batch_size=512, shuffle=True, num_workers=0, 
                                pin_memory=torch.cuda.is_available())

        model = MultiOutputModel(
            input_shape=509,
            hidden_size=3,
            intermediate_layer_1=2,
            intermediate_layer_2=2,
            output_shape_1=3, 
            output_shape_2=15,
            dropout_rate = 0.3
        )
        
        lit_multi_output = LitMultiOuputModel(model, 15, 3)
        early_stop_callback = EarlyStopping(
                monitor='val_f1_score_macro_desc',
                min_delta=0.0,
                patience=10,
                verbose=True,
                mode='max' 
        )
        checkpoint_callback = ModelCheckpoint(
                save_top_k=1,
                monitor="val_f1_score_macro_desc",
                mode="max",
                dirpath=RESULT_PATH + f'model/Adjusted/',
                filename= f'Adjusted_LV{lv}_val{val_fold}_model'
        )
        
        trainer = L.Trainer(callbacks=[early_stop_callback, checkpoint_callback], devices = [device], max_epochs=100, logger=[val_csv_logger])

        trainer.fit(lit_multi_output, tr_loader, val_loader)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SF-RX adjsuted model for DDI Prediction")
    parser.add_argument("--lv", type=int, choices=[1,2,3,4], required=True, help="The level setting")
    parser.add_argument("--device", type=int, required=True, help="GPU device to use for learning")
    
    args = parser.parse_args()
    
    main(args.lv, args.device)