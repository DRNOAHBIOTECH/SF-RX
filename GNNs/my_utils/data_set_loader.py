# %%
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch

import numpy as np
import pandas as pd

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch
import numpy as np

from .prep_ddi import DDIDataProcessor
# %%
class MLPDDIDataSet(Dataset):

    @classmethod
    def set_ddi_info(cls, 
                     ddi_processor: DDIDataProcessor,
                     concat_features: bool = False):
        cls.dict_desc_classes = ddi_processor.dict_desc_classes
        cls.dict_severity_classes = ddi_processor.dict_severity_classes
        cls.dict_action_classes = ddi_processor.dict_action_classes
        cls.NUM_DESC_CLASSES = ddi_processor.NUM_DESC_CLASSES
        cls.NUM_SEVERITY_CLASSES = ddi_processor.NUM_SEVERITY_CLASSES
        cls.NUM_ACTION_CLASSES = ddi_processor.NUM_ACTION_CLASSES
        cls._ddi_info_set = True
        cls.concat_features = concat_features

    def __init__(self, 
                 split_ddi_df: pd.DataFrame, 
                 features_df: pd.DataFrame, 
                 scaler=None):
        super().__init__()
        if not self._ddi_info_set:
            raise ValueError("class method set_ddi_info must be called before creating an instance of MLPDDIDataSet")
        self.split_ddi_df = split_ddi_df
        self.features_df = features_df
        self.scaler = scaler

    def __len__(self):
        return len(self.split_ddi_df)
    
    def __getitem__(self, index):
        ddi_series = self.split_ddi_df.iloc[index]
        subject_drugbank_id = ddi_series["Subject_DrugbankID"]
        affected_drugbank_id = ddi_series["Affected_DrugbankID"]
        severity_str = ddi_series["Severity"]
        desc_str = ddi_series["desc_category"]
        action_str = ddi_series["Action"]

        # X(features)
        subject_tensor = torch.tensor(
            self.features_df.loc[subject_drugbank_id].values.astype(np.float32)
        )
        affected_tensor = torch.tensor(
            self.features_df.loc[affected_drugbank_id].values.astype(np.float32)
        )

        if self.concat_features: # concatenation of features
            concat_tensor = torch.cat([subject_tensor, affected_tensor], dim=-1)
        else: # summation of features
            concat_tensor = subject_tensor + affected_tensor
        # y(labels)
        severity_tensor = self._create_label_tensor(severity_str, self.dict_severity_classes, self.NUM_SEVERITY_CLASSES)
        desc_tensor = self._create_label_tensor(desc_str, self.dict_desc_classes, self.NUM_DESC_CLASSES)
        action_tensor = self._create_label_tensor(action_str, self.dict_action_classes, self.NUM_ACTION_CLASSES)

        return concat_tensor, severity_tensor, desc_tensor, action_tensor

    @staticmethod
    def _create_label_tensor(value, class_dict, num_classes):
        tensor = torch.zeros(num_classes)
        tensor[class_dict[value]] = 1
        return tensor

# %%
class MLPDDIDataSetOpen(Dataset):

    @classmethod
    def set_ddi_info(cls, ddi_processor: DDIDataProcessor):
        cls.dict_desc_classes = ddi_processor.dict_desc_classes
        cls.NUM_DESC_CLASSES = ddi_processor.NUM_DESC_CLASSES
        cls._ddi_info_set = True

    def __init__(self, 
                 split_ddi_df: pd.DataFrame, 
                 features_df: pd.DataFrame, 
                 scaler=None):
        super().__init__()
        if not self._ddi_info_set:
            raise ValueError("class method set_ddi_info must be called before creating an instance of MLPDDIDataSet")
        self.split_ddi_df = split_ddi_df
        self.features_df = features_df
        self.scaler = scaler

    def __len__(self):
        return len(self.split_ddi_df)
    
    def __getitem__(self, index):
        ddi_series = self.split_ddi_df.iloc[index]
        subject_drugbank_id = ddi_series["Subject_DrugbankID"]
        affected_drugbank_id = ddi_series["Affected_DrugbankID"]
        desc_str = ddi_series["desc_category"]

        # X(features)
        subject_tensor = torch.tensor(
            self.features_df.loc[subject_drugbank_id].values.astype(np.float32)
        )
        affected_tensor = torch.tensor(
            self.features_df.loc[affected_drugbank_id].values.astype(np.float32)
        )

        concat_tensor = subject_tensor + affected_tensor

        # y(labels)
        desc_tensor = self._create_label_tensor(desc_str, self.dict_desc_classes, self.NUM_DESC_CLASSES)

        return concat_tensor,  desc_tensor 
    @staticmethod
    def _create_label_tensor(value, class_dict, num_classes):
        tensor = torch.zeros(num_classes)
        tensor[class_dict[value]] = 1
        return tensor