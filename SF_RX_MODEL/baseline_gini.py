import os
import sys
import json
import pickle
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import EarlyStopping

import lightning as L
import torchmetrics

from config import *
from utils import *
sys.path.append("model/")
from baseline import MultiOutputModel

import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('medium')

def main(lv, n_inference, device): 
    X_path = 'data/toy_sum_X.csv'
    y_path = 'data/toy_sum_y.csv'

    X, y = load_data(X_path, y_path)
    temp_df = pd.DataFrame(columns = ['val_fold', 'sev', 'desc', 'dir'])
    for val_fold in range(5):    
        val_X, val_y, _, _ = split_data(X, y, val_fold, lv)
        val_drug_X = torch.tensor(val_X.values, dtype=torch.float32)

        y_val_severity = torch.tensor(pd.get_dummies(val_y.severity).values, dtype=torch.float32)
        y_val_desc = torch.tensor(pd.get_dummies(val_y.type).values, dtype=torch.float32)

        val_dataset = TensorDataset(val_drug_X,
                                    y_val_severity, y_val_desc)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0, 
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

        checkitoput = torch.load(RESULT_PATH + f'model/Baseline/Baseline_LV{lv}_val{val_fold}_model.ckpt')
        state_dict = {k.replace("model.", ""): v for k, v in checkitoput['state_dict'].items()}
        model.load_state_dict(state_dict)

        model.train()
        model = model.to(f'cuda:{device}')

        results_sev = []
        results_desc = []
        for inference in range(n_inference):
            temp_results_sev=[]
            temp_results_desc=[]

            print(inference)
            with torch.no_grad():
                for a, _, _ in val_loader:
                    A = a.to(f'cuda:{device}', non_blocking=True)
                    results_gpu = model(A)
                    temp_results_sev.append(results_gpu[0])
                    temp_results_desc.append(results_gpu[1])

            results_sev.append(torch.cat(temp_results_sev, dim = 0))
            results_desc.append(torch.cat(temp_results_desc, dim = 0))

        with open(RESULT_PATH+f"uncertainty/Baseline/lv{lv}_val{val_fold}_sev_inf{n_inference}_prob.pkl", "wb") as f:
            pickle.dump(results_sev, f)
        with open(RESULT_PATH+f"uncertainty/Baseline/lv{lv}_val{val_fold}_desc_inf{n_inference}_prob.pkl", "wb") as f:
            pickle.dump(results_desc, f)

        sev_res = []
        for infs in range(n_inference):
            sev_res.append(torch.argmax(results_sev[infs], dim = 1))

        desc_res = []
        for infs in range(n_inference):
            desc_res.append(torch.argmax(results_desc[infs], dim = 1))

        sev_ginis = load_preds(sev_res, 'sev')
        desc_ginis = load_preds(desc_res, 'desc')

        plt.hist(sev_ginis)
        plt.title(f'[Level {lv}, Validation {val_fold}, DDI Severity]: Historgram of Gini Impurity')
        plt.xlabel('Gini Impurity')
        plt.ylabel('Frequency')
        plt.savefig(RESULT_PATH + f"uncertainty/Baseline/lv{lv}_Gini_Hist_Severity_{val_fold}_inf{n_inference}.png")
        plt.clf()

        plt.hist(desc_ginis)
        plt.title(f'[Level {lv}, Validation {val_fold}, DDI Type]: Historgram of Gini Impurity')
        plt.xlabel('Gini Impurity')
        plt.ylabel('Frequency')
        plt.savefig(RESULT_PATH + f"uncertainty/Baseline/lv{lv}_Gini_Hist_Description_{val_fold}_inf{n_inference}.png") 
        plt.clf()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SF-RX baseline model for DDI Prediction")
    parser.add_argument("--lv", type=int, choices=[1,2,3,4], required=True, help="The level setting")
    parser.add_argument("--n_inference", type=int, required=True, help="Number of inference for calculating Gini Impurity")
    parser.add_argument("--device", type=int, required=True, help="GPU device to use for learning")
    
    args = parser.parse_args()
    
    main(args.lv, args.n_inference, args.device)