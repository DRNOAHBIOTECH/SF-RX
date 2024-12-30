import os
import sys
import torch
import numpy as np
import pandas as pd
from torchmetrics.classification import *
from torcheval.metrics import MulticlassAUPRC
from config import DATA_PATH

def data_splitter(lv):
    def lv1_split(ddi_info, ts_fold):
        ts = ddi_info.lv1 == ts_fold
        tr = ~ts
        return tr, ts

    def lv2_split(ddi_info, ts_fold):
        ts = ddi_info.fold_id_1 == ts_fold
        tr = ~ts
        return tr, ts

    def lv3_split(ddi_info, ts_fold):
        ts = (ddi_info.fold_id_1 == ts_fold) ^ (ddi_info.fold_id_2 == ts_fold)
        tr = ~ts & ~((ddi_info.fold_id_1 == ts_fold) & (ddi_info.fold_id_2 == ts_fold))
        return tr, ts

    def lv4_split(ddi_info, ts_fold):
        ts = (ddi_info.fold_id_1 == ts_fold) & (ddi_info.fold_id_2 == ts_fold)
        tr = ~ts & ~((ddi_info.fold_id_1 == ts_fold) ^ (ddi_info.fold_id_2 == ts_fold))
        return tr, ts
    
    if lv == 1:
        data_split = lv1_split
    elif lv == 2:
        data_split = lv2_split
    elif lv == 3:
        data_split = lv3_split
    else: # lv == 4
        data_split = lv4_split
        
    return data_split

def data_loading(dat_src):
    y = pd.read_parquet(DATA_PATH + f'{dat_src}_y.parquet')
    y_sev = pd.read_parquet(DATA_PATH + f'{dat_src}_y_sev.parquet')
    X = pd.read_parquet(DATA_PATH + f'{dat_src}_X.parquet')
    
    return X, y, y_sev

def load_model_parts(model, share_ckpt, output_ckpt):
    model.norm.load_state_dict(share_ckpt['norm'], strict=False)
    model.hidden_layers.load_state_dict(share_ckpt['hidden_layers'], strict=False)
    model.intermediate.load_state_dict(share_ckpt['intermediate'], strict=False)
    
    model.output_1.load_state_dict(output_ckpt['output_1'], strict=False)
    print("Model parts have been successfully loaded.")

def calculate_metrics_sev(
    y_true_severity: torch.Tensor, 
    y_pred_severity: torch.Tensor, 
    num_classes_severity: int
) -> dict[str, float]:
    
    y_true_severity = y_true_severity.clone().detach().long()
    y_pred_severity = y_pred_severity.clone().detach().float()
    
    y_true_severity2d = y_true_severity
    y_pred_severity2d = y_pred_severity
    
    f1_macro_severity = MulticlassF1Score(num_classes=num_classes_severity, average='macro')
    f1_micro_severity = MulticlassF1Score(num_classes=num_classes_severity, average='micro')
    recall_macro_severity = MulticlassRecall(num_classes=num_classes_severity, average='macro')
    recall_micro_severity = MulticlassRecall(num_classes=num_classes_severity, average='micro')
    precision_macro_severity = MulticlassPrecision(num_classes=num_classes_severity, average='macro')
    precision_micro_severity = MulticlassPrecision(num_classes=num_classes_severity, average='micro')
    auc_pr_severity = MulticlassAUPRC(num_classes=num_classes_severity)
    auc_roc_severity = MulticlassAUROC(num_classes=num_classes_severity)
    accuracy_macro_severity = MulticlassAccuracy(num_classes=num_classes_severity, average='macro')
    accuracy_micro_severity = MulticlassAccuracy(num_classes=num_classes_severity, average='micro')
    def aucpr(pred, targ):
        auc_pr_severity.update(pred, targ)
        return auc_pr_severity.compute().item()
    def aucroc(pred, targ):
        auc_roc_severity.update(pred, targ)
        return auc_roc_severity.compute().item()
    
    y_pred_severity = np.argmax(y_pred_severity, axis = 1)
    y_true_severity = np.argmax(y_true_severity, axis = 1)
    
    metrics_severity = {
        'f1_macro': f1_macro_severity(y_pred_severity, y_true_severity).item(),
        'f1_micro': f1_micro_severity(y_pred_severity, y_true_severity).item(),
        'recall_macro': recall_micro_severity(y_pred_severity, y_true_severity).item(),
        'recall_micro': recall_micro_severity(y_pred_severity, y_true_severity).item(),
        'precision_macro': precision_macro_severity(y_pred_severity, y_true_severity).item(),
        'precision_micro': precision_micro_severity(y_pred_severity, y_true_severity).item(),
        'auc_pr': aucpr(y_pred_severity2d, y_true_severity),
        'auc_roc': aucroc(y_pred_severity2d, y_true_severity),
        'accuracy_macro': accuracy_macro_severity(y_pred_severity, y_true_severity).item(),
        'accuracy_micro': accuracy_micro_severity(y_pred_severity, y_true_severity).item()
    }
    
    return metrics_severity

def metric_output_sev(predictions: list[torch.Tensor], true_severity: torch.Tensor) -> dict[str, float]:
    pred_severity = []

    for sev in predictions:
        pred_severity.append(sev)

    pred_severity = torch.cat(pred_severity, dim=0)

    metrics = calculate_metrics_sev(true_severity, pred_severity, true_severity.shape[1])

    return metrics