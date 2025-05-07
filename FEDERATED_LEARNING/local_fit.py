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
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping

import lightning as L
import torchmetrics

from fl_utils import data_splitter, data_loading, load_model_parts, metric_output_sev
from model import MlpMultiTask, LitMlpMultiTask, load_hyperparam
from config import *

torch.set_float32_matmul_precision('medium')

def main(dat_src, lv, fl_round, device):    
    rsts = RESULT_PATH + f'lv{lv}/round{fl_round}/'
    rsts_1 = RESULT_PATH + f'lv{lv}/round{fl_round-1}/'
    
    X, y, y_sev = data_loading(dat_src)
    
    data_split = data_splitter(lv)

    te_fold = TE_FOLD[dat_src]
    val_fold = VAL_FOLD[dat_src]
    
    val_logger = TensorBoardLogger(save_dir=RESULT_PATH + f'log/lv{lv}/', name=f"{dat_src}", version=f'Round{fl_round}')
    
    MD_idx, te_idx = data_split(y, te_fold)
    te_X, te_y_sev = X[te_idx].astype(np.float32), y_sev[te_idx].astype(np.float32)
    MD_X, MD_y_sev, MD_y = X[MD_idx], y_sev[MD_idx], y[MD_idx]
    
    tr_idx, _ = data_split(MD_y, val_fold)
    val_idx = ~tr_idx
    
    tr_X, tr_y_sev = MD_X[tr_idx].astype(np.float32), MD_y_sev[tr_idx].astype(np.float32)
    val_X, val_y_sev = MD_X[val_idx].astype(np.float32), MD_y_sev[val_idx].astype(np.float32)
    
    y_tr_true_sev = torch.tensor(tr_y_sev.values, dtype=torch.float32).cpu()
    y_val_true_sev = torch.tensor(val_y_sev.values, dtype=torch.float32).cpu()
    y_te_true_sev = torch.tensor(te_y_sev.values, dtype=torch.float32).cpu()

    train_dataset = TensorDataset(torch.tensor(tr_X.values, dtype=torch.float32), torch.tensor(tr_y_sev.values, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=torch.cuda.is_available())
    
    val_dataset = TensorDataset(torch.tensor(val_X.values, dtype=torch.float32), torch.tensor(val_y_sev.values, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=torch.cuda.is_available())

    te_dataset = TensorDataset(torch.tensor(te_X.values, dtype=torch.float32), torch.tensor(te_y_sev.values, dtype=torch.float32))
    te_loader = DataLoader(te_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=torch.cuda.is_available())
    
    # load avg model
    share_ckpt = torch.load(rsts_1 + 'shared_avg_layers.pth')
    output_ckpt = torch.load(rsts_1 + f'{dat_src}_output_layers.pth')
    
    model = MlpMultiTask(**load_hyperparam(lv, dat_src))
    load_model_parts(model, share_ckpt, output_ckpt)
    lit_multi_output = LitMlpMultiTask(model, OUTOUT_SHAPE[dat_src], 0)
        
    early_stop_callback = EarlyStopping(
        monitor='sev_validation_f1_macro',
        min_delta=0.0,
        patience=10,
        verbose=True,
        mode='max' 
    )

    trainer = L.Trainer(callbacks=[early_stop_callback], devices=[device], max_epochs=100, logger=val_logger)
    trainer.fit(lit_multi_output, train_loader, val_loader)
    stopped_epoch = early_stop_callback.stopped_epoch
    torch.save(model.state_dict(), rsts + f'{dat_src}_model.pth')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=torch.cuda.is_available())
    tr_predictions = trainer.predict(lit_multi_output, train_loader)
    val_predictions = trainer.predict(lit_multi_output, val_loader)
    te_predictions = trainer.predict(lit_multi_output, te_loader)
    
    train_results = metric_output_sev(tr_predictions, y_tr_true_sev)
    val_results = metric_output_sev(val_predictions, y_val_true_sev)
    test_results = metric_output_sev(te_predictions, y_te_true_sev)
    
    result1 = pd.DataFrame([train_results, val_results, test_results])
    result1.index = ['train', 'validation', 'test']
    result1.to_csv(rsts + f'{dat_src}_results.csv')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mlp for DDI Prediction")
    parser.add_argument("--dat_src", type=str, choices=['drugscom', 'pdr'], required=True, help="The data source to use")
    parser.add_argument("--lv", type=int, choices=[1,2,3,4], required=True, help="The level setting")
    parser.add_argument("--fl_round", type=int, required=True, help="Round for federated learnings")
    parser.add_argument("--device", type=int, choices=[0, 1, 2, 3], required=True, help="The gpu device to use")

    args = parser.parse_args()
    
    main(args.dat_src, args.lv, args.fl_round, args.device)