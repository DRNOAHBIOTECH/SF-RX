from typing import List, Dict, Union, Callable
import copy

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
import torch
import pandas as pd
from torch_geometric.data import Data, Dataset
from .prep_ddi import DDIDataProcessor
import torch_geometric.utils.smiles as PyG_smiles
from .dce_prepare_smiles import smiles_to_dce_features

class DDI_GCN_Dataset(Dataset):

    @classmethod
    def set_ddi_info(cls, ddi_processor:DDIDataProcessor):
        cls.dict_desc_classes = ddi_processor.dict_desc_classes
        cls.dict_severity_classes = ddi_processor.dict_severity_classes
        cls.NUM_DESC_CLASSES = ddi_processor.NUM_DESC_CLASSES
        cls.NUM_SEVERITY_CLASSES = ddi_processor.NUM_SEVERITY_CLASSES
        cls.db_id_to_smiles_dict = ddi_processor.db_id_to_smiles_dict
        cls._ddi_info_set = True 
        cls.pyg_data_dict = {}
        for db_id in ddi_processor.db_id_set_for_ddi:
            data = PyG_smiles.from_smiles(ddi_processor.db_id_to_smiles_dict[db_id])
            cls.pyg_data_dict[db_id] = data

    def __init__(self, split_df):
        super().__init__()
        if not self._ddi_info_set:
            raise ValueError("class method set_ddi_info must be called before creating an instance of DDI_Graph_Dataset")
        self.split_df = split_df

    def len(self):
        return len(self.split_df)

    def get(self, idx):
        row_ddi = self.split_df.iloc[idx]
        data_combined_ddi = self._create_combined_separate_drug_graph_data(row_ddi)
        return data_combined_ddi

    def _create_combined_separate_drug_graph_data(self, row: pd.Series) -> Data:
        sub_id, aff_id = row.Subject_DrugbankID, row.Affected_DrugbankID
        severity_str, desc_str  = row.Severity, row.type
        sub_pyg_data = self.pyg_data_dict[sub_id]
        aff_pyg_data = self.pyg_data_dict[aff_id]

        # Combine two drugs (two separate graphs) into one graph
        combined_edge_index = torch.cat(
            (sub_pyg_data.edge_index,
            aff_pyg_data.edge_index + sub_pyg_data.x.size(0)), dim=1
        )
        sub_node_attr = smiles_to_dce_features(self.db_id_to_smiles_dict[sub_id])[0]
        aff_node_attr = smiles_to_dce_features(self.db_id_to_smiles_dict[aff_id])[0]
        combined_node_attr = torch.cat((sub_node_attr, aff_node_attr), dim=0)
        combined_edge_attr = torch.cat((sub_pyg_data.edge_attr, 
                                        aff_pyg_data.edge_attr), dim=0)

        data_combined = Data(x=combined_node_attr, 
                             edge_index=combined_edge_index, 
                             edge_attr=combined_edge_attr)
        data_combined.sub_id = sub_id
        data_combined.aff_id = aff_id

        severity_tensor = torch.zeros(self.NUM_SEVERITY_CLASSES)
        severity_tensor[self.dict_severity_classes[severity_str]] = 1
        desc_tensor = torch.zeros(self.NUM_DESC_CLASSES)
        desc_tensor[self.dict_desc_classes[desc_str]] = 1
        data_combined.severity_y = severity_tensor.unsqueeze(0)
        data_combined.desc_y = desc_tensor.unsqueeze(0)

        return data_combined

class GNN_DataSplitterByLevel:
    def __init__(self, 
                 ddi_label_df: pd.DataFrame, 
                 DDI_Graph_Dataset: Dataset,
                 n_splits: int = 5, 
                 batch_size: int = 256, 
                 num_workers: int = 4):
        self.ddi_label_df = ddi_label_df
        self.PYG_DATA_DICT = DDI_Graph_Dataset.pyg_data_dict
        self.DDI_Graph_Dataset = DDI_Graph_Dataset
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.num_workers = num_workers

    def split_data(self, method: str = 'v1') -> List[Dict[str, DataLoader]]:
        if method == 'v1':
            return self._split_data_v1()
        elif method == 'v2':
            return self._split_data_v2()
        elif method == 'v2_1':
            return self._split_data_v2_1()
        elif method == 'v3':
            return self._split_data_v3()
        elif method == 'v4':
            return self._split_data_v4()
        else:
            raise ValueError("Invalid method. Choose 'v1', 'v2','v2_1', 'v3', or 'v4'.")

    def _split_data_v1(self) -> List[Dict[str, DataLoader]]:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        split_idx: List[np.ndarray] = []
        for _, val_index in skf.split(self.ddi_label_df, self.ddi_label_df['desc_category_integer']):
            split_idx.append(val_index)
        
        fold_dl: List[Dict[str, DataLoader]] = []
        for fold_idx in range(self.n_splits):
            copied_split_idx = copy.deepcopy(split_idx)
            val_idx = copied_split_idx.pop(fold_idx)
            train_idx = np.concatenate(copied_split_idx)
            
            fold_dict: Dict[str, DataLoader] = {}
            ddi_val_df = self.ddi_label_df.iloc[val_idx]
            ddi_val_dt = self.DDI_Graph_Dataset(ddi_val_df)
            fold_dict['val'] = DataLoader(ddi_val_dt, batch_size=self.batch_size, 
                                          shuffle=False, num_workers=self.num_workers)
            
            ddi_train_df = self.ddi_label_df.iloc[train_idx]
            ddi_train_dt = self.DDI_Graph_Dataset(ddi_train_df)
            fold_dict['train'] = DataLoader(ddi_train_dt, batch_size=self.batch_size,
                                            shuffle=True, num_workers=self.num_workers)
            
            fold_dl.append(fold_dict)
        
        return fold_dl

    def _split_data_v2(self) -> List[Dict[str, DataLoader]]:
        fold_dl: List[Dict[str, DataLoader]] = []
        for fold_idx in range(self.n_splits):
            fold_dict: Dict[str, DataLoader] = {}
            fold_dict['val'] = DataLoader(
                self.DDI_Graph_Dataset(
                    self.ddi_label_df.query(f"subject_fold== {fold_idx}")), 
                    batch_size=self.batch_size, shuffle=False, 
                    num_workers=self.num_workers)
            
            train_fold_idx = set(range(self.n_splits)) - set([fold_idx])
            fold_dict['train'] = DataLoader(
                self.DDI_Graph_Dataset(
                    self.ddi_label_df[self.ddi_label_df['subject_fold'].isin(train_fold_idx)]), 
                    batch_size=self.batch_size, shuffle=True, 
                    num_workers=self.num_workers)
            
            fold_dl.append(fold_dict)
        
        return fold_dl

    def _split_data_v2_1(self, n_splits=5) -> List[Dict[str, DataLoader]]:
        fold_dl = []
        for fold_idx in range(n_splits):
            fold_dict = {}
            # DataFrame
            ddi_val_df = self.ddi_label_df.query(
                f"affected_fold == {fold_idx}")
            train_fold_idx = set(range(n_splits)) - set([fold_idx])
            ddi_train_df = self.ddi_label_df[
                self.ddi_label_df['affected_fold'].isin(train_fold_idx)]

            fold_dict['val'] = DataLoader(
                    self.DDI_Graph_Dataset(ddi_val_df),
                    batch_size=self.batch_size, shuffle=False, 
                    num_workers=self.num_workers)

            fold_dict['train'] = DataLoader(
                    self.DDI_Graph_Dataset(ddi_train_df),
                    batch_size=self.batch_size, shuffle=True, 
                    num_workers=self.num_workers)

            fold_dl.append(fold_dict)
        
        return fold_dl

    def _split_data_v3(self) -> List[Dict[str, DataLoader]]:
        fold_dl: List[Dict[str, DataLoader]] = []
        for fold_idx in range(self.n_splits):
            test_fold_bool = (self.ddi_label_df['subject_fold'] == fold_idx) | (self.ddi_label_df['affected_fold'] == fold_idx)
            ddi_valid_fold = self.ddi_label_df[test_fold_bool]
            ddi_valid_fold = ddi_valid_fold[
                ddi_valid_fold.subject_fold != ddi_valid_fold.affected_fold
            ]
            ddi_train_fold = self.ddi_label_df[~test_fold_bool]
            
            fold_dict: Dict[str, DataLoader] = {}
            fold_dict['val'] = DataLoader(
                self.DDI_Graph_Dataset(ddi_valid_fold), 
                batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            
            fold_dict['train'] = DataLoader(
                self.DDI_Graph_Dataset(ddi_train_fold), 
                batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            
            fold_dl.append(fold_dict)
        
        return fold_dl

    def _split_data_v4(self) -> List[Dict[str, DataLoader]]:
        fold_dl: List[Dict[str, DataLoader]] = []
        for fold_idx in range(self.n_splits):
            test_fold_bool = (self.ddi_label_df['subject_fold'] == fold_idx) & (self.ddi_label_df['affected_fold'] == fold_idx)
            list_folds = list(range(self.n_splits))
            list_folds.pop(fold_idx)
            train_fold_bool = (self.ddi_label_df['subject_fold'].isin(list_folds)) & (self.ddi_label_df['affected_fold'].isin(list_folds))
            
            ddi_valid_fold = self.ddi_label_df[test_fold_bool]
            ddi_train_fold = self.ddi_label_df[train_fold_bool]

            fold_dict: Dict[str, DataLoader] = {}
            fold_dict['val'] = DataLoader(
                self.DDI_Graph_Dataset(ddi_valid_fold), 
                batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            
            fold_dict['train'] = DataLoader(
                self.DDI_Graph_Dataset(ddi_train_fold), 
                batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            
            fold_dl.append(fold_dict)
        
        return fold_dl


class DDI_Hybrid_Dataset(Dataset):

    @classmethod
    def set_ddi_info(cls, ddi_processor:DDIDataProcessor):
        cls.dict_desc_classes = ddi_processor.dict_desc_classes
        cls.dict_severity_classes = ddi_processor.dict_severity_classes
        cls.dict_action_classes = ddi_processor.dict_action_classes
        cls.NUM_DESC_CLASSES = ddi_processor.NUM_DESC_CLASSES
        cls.NUM_SEVERITY_CLASSES = ddi_processor.NUM_SEVERITY_CLASSES
        cls.NUM_ACTION_CLASSES = ddi_processor.NUM_ACTION_CLASSES
        cls.db_id_to_smiles_dict = ddi_processor.db_id_to_smiles_dict
        cls._ddi_info_set = True 
        cls.pyg_data_dict = {}
        for db_id in ddi_processor.db_id_set_for_ddi:
            data = PyG_smiles.from_smiles(ddi_processor.db_id_to_smiles_dict[db_id])
            cls.pyg_data_dict[db_id] = data

    def __init__(self, split_df, features_df):
        super().__init__()
        if not self._ddi_info_set:
            raise ValueError("class method set_ddi_info must be called before creating an instance of DDI_Graph_Dataset")
        self.split_df = split_df
        self.features_df = features_df
    def len(self):
        return len(self.split_df)
    def get(self, idx):
        ddi_series = self.split_df.iloc[idx]
        # Create MLP data
        subject_drugbank_id = ddi_series["Subject_DrugbankID"]
        affected_drugbank_id = ddi_series["Affected_DrugbankID"]

        # X(features)
        subject_tensor =\
            torch.tensor(
                self.features_df
                .loc[subject_drugbank_id]
                .values.astype(np.float32)
            )
        affected_tensor =\
            torch.tensor(
                self.features_df
                .loc[affected_drugbank_id]
                .values.astype(np.float32)
            )
        concat_tensor = subject_tensor + affected_tensor

        # Create the combined graph data
        data_combined_ddi = self._create_combined_separate_drug_graph_data(
            ddi_series)
        return concat_tensor, data_combined_ddi

    def _create_combined_separate_drug_graph_data(self, row: pd.Series) -> Data:
        sub_id, aff_id = row.Subject_DrugbankID, row.Affected_DrugbankID
        severity_str, desc_str, action_str = row.Severity, row.desc_category, row.Action
        sub_pyg_data = self.pyg_data_dict[sub_id]
        aff_pyg_data = self.pyg_data_dict[aff_id]

        # Combine two drugs (two separate graphs) into one graph
        combined_edge_index = torch.cat(
            (sub_pyg_data.edge_index,
            aff_pyg_data.edge_index + sub_pyg_data.x.size(0)), dim=1
        )
        sub_node_attr = smiles_to_dce_features(self.db_id_to_smiles_dict[sub_id])[0]
        aff_node_attr = smiles_to_dce_features(self.db_id_to_smiles_dict[aff_id])[0]
        combined_node_attr = torch.cat((sub_node_attr, aff_node_attr), dim=0)
        combined_edge_attr = torch.cat((sub_pyg_data.edge_attr, 
                                        aff_pyg_data.edge_attr), dim=0)

        data_combined = Data(x=combined_node_attr, 
                             edge_index=combined_edge_index, 
                             edge_attr=combined_edge_attr)
        data_combined.sub_id = sub_id
        data_combined.aff_id = aff_id

        severity_tensor = torch.zeros(self.NUM_SEVERITY_CLASSES)
        severity_tensor[self.dict_severity_classes[severity_str]] = 1
        desc_tensor = torch.zeros(self.NUM_DESC_CLASSES)
        desc_tensor[self.dict_desc_classes[desc_str]] = 1
        action_tensor = torch.zeros(self.NUM_ACTION_CLASSES)
        action_tensor[self.dict_action_classes[action_str]] = 1
        data_combined.severity_y = severity_tensor.unsqueeze(0)
        data_combined.desc_y = desc_tensor.unsqueeze(0)
        data_combined.action_y = action_tensor.unsqueeze(0)

        return data_combined
class Hybrid_DataSplitterByLevel:
    def __init__(self, 
                 ddi_label_df: pd.DataFrame, 
                 feature_df: pd.DataFrame, 
                 PYG_DATA_DICT: Dict,
                 DDI_Graph_Dataset: Callable,
                 n_splits: int = 5, 
                 batch_size: int = 256, 
                 num_workers: int = 32):
        self.ddi_label_df = ddi_label_df
        self.feature_df = feature_df
        self.PYG_DATA_DICT = PYG_DATA_DICT
        self.DDI_Graph_Dataset = DDI_Graph_Dataset
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.num_workers = num_workers

    def split_data(self, method: str = 'v1') -> List[Dict[str, DataLoader]]:
        if method == 'v1':
            return self._split_data_v1()
        elif method == 'v2':
            return self._split_data_v2()
        elif method == 'v3':
            return self._split_data_v3()
        elif method == 'v4':
            return self._split_data_v4()
        else:
            raise ValueError("Invalid method. Choose 'v1', 'v2', 'v3', or 'v4'.")

    def _split_data_v1(self) -> List[Dict[str, DataLoader]]:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        split_idx: List[np.ndarray] = []
        for _, val_index in skf.split(self.ddi_label_df, self.ddi_label_df['desc_category_integer']):
            split_idx.append(val_index)
        
        fold_dl: List[Dict[str, DataLoader]] = []
        for fold_idx in range(self.n_splits):
            copied_split_idx = copy.deepcopy(split_idx)
            val_idx = copied_split_idx.pop(fold_idx)
            train_idx = np.concatenate(copied_split_idx)
            
            fold_dict: Dict[str, DataLoader] = {}
            ddi_val_df = self.ddi_label_df.iloc[val_idx]
            ddi_val_dt = self.DDI_Graph_Dataset(
                ddi_val_df, self.feature_df)
            fold_dict['val'] = DataLoader(ddi_val_dt, batch_size=self.batch_size, 
                                          shuffle=False, num_workers=self.num_workers)
            
            ddi_train_df = self.ddi_label_df.iloc[train_idx]
            ddi_train_dt = self.DDI_Graph_Dataset(
                ddi_train_df, self.feature_df)
            fold_dict['train'] = DataLoader(ddi_train_dt, batch_size=self.batch_size,
                                            shuffle=True, num_workers=self.num_workers)
            
            fold_dl.append(fold_dict)
        
        return fold_dl

    def _split_data_v2(self) -> List[Dict[str, DataLoader]]:
        fold_dl: List[Dict[str, DataLoader]] = []
        for fold_idx in range(self.n_splits):
            fold_dict: Dict[str, DataLoader] = {}
            fold_dict['val'] = DataLoader(
                self.DDI_Graph_Dataset(
                    self.ddi_label_df.query(f"subject_fold== {fold_idx}"), 
                    self.feature_df,
                ), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            
            train_fold_idx = set(range(self.n_splits)) - set([fold_idx])
            fold_dict['train'] = DataLoader(
                self.DDI_Graph_Dataset(
                    self.ddi_label_df[self.ddi_label_df['subject_fold'].isin(train_fold_idx)],
                    self.feature_df,
                ), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            
            fold_dl.append(fold_dict)
        
        return fold_dl

    def _split_data_v3(self) -> List[Dict[str, DataLoader]]:
        fold_dl: List[Dict[str, DataLoader]] = []
        for fold_idx in range(self.n_splits):
            test_fold_bool = (self.ddi_label_df['subject_fold'] == fold_idx) | (self.ddi_label_df['affected_fold'] == fold_idx)
            ddi_valid_fold = self.ddi_label_df[test_fold_bool]
            ddi_train_fold = self.ddi_label_df[~test_fold_bool]
            
            fold_dict: Dict[str, DataLoader] = {}
            fold_dict['val'] = DataLoader(
                self.DDI_Graph_Dataset(
                    ddi_valid_fold,
                    self.feature_df,), 
                batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            
            fold_dict['train'] = DataLoader(
                self.DDI_Graph_Dataset(
                    ddi_train_fold,
                    self.feature_df,), 
                batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            
            fold_dl.append(fold_dict)
        
        return fold_dl

    def _split_data_v4(self) -> List[Dict[str, DataLoader]]:
        fold_dl: List[Dict[str, DataLoader]] = []
        for fold_idx in range(self.n_splits):
            test_fold_bool = (self.ddi_label_df['subject_fold'] == fold_idx) & (self.ddi_label_df['affected_fold'] == fold_idx)
            list_folds = list(range(self.n_splits))
            list_folds.pop(fold_idx)
            train_fold_bool = (self.ddi_label_df['subject_fold'].isin(list_folds)) & (self.ddi_label_df['affected_fold'].isin(list_folds))
            
            ddi_valid_fold = self.ddi_label_df[test_fold_bool]
            ddi_train_fold = self.ddi_label_df[train_fold_bool]

            fold_dict: Dict[str, DataLoader] = {}
            fold_dict['val'] = DataLoader(
                self.DDI_Graph_Dataset(
                    ddi_valid_fold,
                    self.feature_df,), 
                batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
            
            fold_dict['train'] = DataLoader(
                self.DDI_Graph_Dataset(
                    ddi_train_fold,
                    self.feature_df,), 
                batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            
            fold_dl.append(fold_dict)
        
        return fold_dl



class DDI_GCN_DatasetOpen(Dataset):

    @classmethod
    def set_ddi_info(cls, ddi_processor:DDIDataProcessor):
        cls.dict_desc_classes = ddi_processor.dict_desc_classes
        cls.NUM_DESC_CLASSES = ddi_processor.NUM_DESC_CLASSES
        cls.db_id_to_smiles_dict = ddi_processor.db_id_to_smiles_dict
        cls._ddi_info_set = True 
        cls.pyg_data_dict = {}
        for db_id in ddi_processor.db_id_set_for_ddi:
            data = PyG_smiles.from_smiles(ddi_processor.db_id_to_smiles_dict[db_id])
            cls.pyg_data_dict[db_id] = data

    def __init__(self, split_df):
        super().__init__()
        if not self._ddi_info_set:
            raise ValueError("class method set_ddi_info must be called before creating an instance of DDI_Graph_Dataset")
        self.split_df = split_df

    def len(self):
        return len(self.split_df)

    def get(self, idx):
        row_ddi = self.split_df.iloc[idx]
        data_combined_ddi = self._create_combined_separate_drug_graph_data(row_ddi)
        return data_combined_ddi

    def _create_combined_separate_drug_graph_data(self, row: pd.Series) -> Data:
        sub_id, aff_id = row.Subject_DrugbankID, row.Affected_DrugbankID
        desc_str =  row.desc_category
        sub_pyg_data = self.pyg_data_dict[sub_id]
        aff_pyg_data = self.pyg_data_dict[aff_id]

        # Combine two drugs (two separate graphs) into one graph
        combined_edge_index = torch.cat(
            (sub_pyg_data.edge_index,
            aff_pyg_data.edge_index + sub_pyg_data.x.size(0)), dim=1
        )
        sub_node_attr = smiles_to_dce_features(self.db_id_to_smiles_dict[sub_id])[0]
        aff_node_attr = smiles_to_dce_features(self.db_id_to_smiles_dict[aff_id])[0]
        combined_node_attr = torch.cat((sub_node_attr, aff_node_attr), dim=0)
        combined_edge_attr = torch.cat((sub_pyg_data.edge_attr, 
                                        aff_pyg_data.edge_attr), dim=0)

        data_combined = Data(x=combined_node_attr, 
                             edge_index=combined_edge_index, 
                             edge_attr=combined_edge_attr)
        data_combined.sub_id = sub_id
        data_combined.aff_id = aff_id

        desc_tensor = torch.zeros(self.NUM_DESC_CLASSES)
        desc_tensor[self.dict_desc_classes[desc_str]] = 1
        data_combined.desc_y = desc_tensor.unsqueeze(0)

        return data_combined