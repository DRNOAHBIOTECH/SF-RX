import pandas as pd
import numpy as np
import torch
import copy

from collections import OrderedDict
from model import MlpMultiTask, load_hyperparam
from config import *

np.random.seed(SEED)
torch.manual_seed(SEED)

def split_model(model):
    shared_layers = {
        'norm': model.norm.state_dict(),
        'hidden_layers': model.hidden_layers.state_dict(),
        'intermediate': model.intermediate.state_dict(),
    }
    output_layers = {
        'output_1': model.output_1.state_dict()
    }
    return shared_layers, output_layers

def average_shared_weights(shared_layers_list):
    avg_weights = copy.deepcopy(shared_layers_list[0])
    
    num_models = len(shared_layers_list)
    
    for layer_key in avg_weights.keys():
        for weight_key in avg_weights[layer_key]:
            weight_sum = sum(shared_layers[layer_key][weight_key] for shared_layers in shared_layers_list)
            avg_weights[layer_key][weight_key] = weight_sum / num_models
    
    return avg_weights

def model_initialization(lv, fl_round):
    rsts = RESULT_PATH + f'lv{lv}/round{fl_round}/'
    models = []
    for dat_src in DATA_SOURCE:
        model = MlpMultiTask(**load_hyperparam(lv, dat_src))
        model.load_state_dict(torch.load(rsts + f'{dat_src}_model.pth'))
        share_layers, output_layers = split_model(model)
        models.append(share_layers)
        torch.save(output_layers, rsts + f'{dat_src}_output_layers.pth')
    avg_share_layers = average_shared_weights(models)
    torch.save(avg_share_layers, rsts + f'shared_avg_layers.pth')
    
def generate_null_models(lv): # for the round 0
    rsts = RESULT_PATH + f'lv{lv}/round{0}/'
    for dat_src in DATA_SOURCE:
        model = MlpMultiTask(**load_hyperparam(lv, dat_src))
        torch.save(model.state_dict(), rsts + f'{dat_src}_model.pth')
