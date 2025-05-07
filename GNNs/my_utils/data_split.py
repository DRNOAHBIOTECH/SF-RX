import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

class DataSplitterByLevel:
    def __init__(self, ddi_label_df, 
                 features_df, 
                 DDIDataSet, 
                 batch_size,
                 concat_features,
                 num_workers):
        self.ddi_label_df = ddi_label_df
        self.features_df = features_df
        self.DDIDataSet = DDIDataSet
        self.batch_size = batch_size
        self.concat_features = concat_features
        self.num_workers = num_workers

    def _split_method_1(self, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_idx = []
        for _, (_, val_index) in enumerate(skf.split(self.ddi_label_df, self.ddi_label_df['desc_category_integer'])):
            split_idx.append(val_index)

        fold_dl = []
        for fold_idx in range(n_splits):
            copied_split_idx = copy.deepcopy(split_idx)
            val_idx = copied_split_idx.pop(fold_idx)
            train_idx = np.concatenate(copied_split_idx)
            fold_dict = {}
            # DataFrame
            ddi_val_df = self.ddi_label_df.iloc[val_idx]
            ddi_train_df = self.ddi_label_df.iloc[train_idx]

            # Concatenate the features
            if self.concat_features:
                ddi_train_df = pd.concat(
                [
                    ddi_train_df,
                    ddi_train_df.rename(
                        columns={"Subject_DrugbankID": "Affected_DrugbankID", 
                                "Affected_DrugbankID": "Subject_DrugbankID"})
                ],
                    axis= 0
                )
                ddi_val_df = pd.concat(
                [
                    ddi_val_df,
                    ddi_val_df.rename(
                        columns={"Subject_DrugbankID": "Affected_DrugbankID", 
                                "Affected_DrugbankID": "Subject_DrugbankID"})
                ],      
                    axis= 0
                )
            # Dataset
            ddi_train_dt = self.DDIDataSet(ddi_train_df, self.features_df)
            ddi_val_dt = self.DDIDataSet(ddi_val_df, self.features_df)

            # DataLoader
            fold_dict['val'] = DataLoader(
                ddi_val_dt,
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=self.num_workers)

            fold_dict['train'] = DataLoader(
                ddi_train_dt,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            fold_dl.append(fold_dict)
        
        return fold_dl

    def _split_method_2(self, n_splits=5):
        fold_dl = []
        for fold_idx in range(n_splits):
            fold_dict = {}
            # DataFrame
            ddi_val_df = self.ddi_label_df.query(
                f"subject_fold == {fold_idx}")
            train_fold_idx = set(range(n_splits)) - set([fold_idx])
            ddi_train_df = self.ddi_label_df[
                self.ddi_label_df['subject_fold'].isin(train_fold_idx)]

            # Concatenate the features
            if self.concat_features:
                ddi_train_df = pd.concat(
                [
                    ddi_train_df,
                    ddi_train_df.rename(
                        columns={"Subject_DrugbankID": "Affected_DrugbankID", 
                                "Affected_DrugbankID": "Subject_DrugbankID"})
                ],
                    axis= 0
                )
                ddi_val_df = pd.concat(
                [
                    ddi_val_df,
                    ddi_val_df.rename(
                        columns={"Subject_DrugbankID": "Affected_DrugbankID", 
                                "Affected_DrugbankID": "Subject_DrugbankID"})
                ],      
                    axis= 0
                )
            ddi_train_dt = self.DDIDataSet(ddi_train_df, self.features_df)
            ddi_val_dt = self.DDIDataSet(ddi_val_df, self.features_df)

            # DataLoader
            fold_dict['val'] = DataLoader(
                ddi_val_dt,
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=self.num_workers)

            fold_dict['train'] = DataLoader(
                ddi_train_dt,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            fold_dl.append(fold_dict)
        
        return fold_dl

    def _split_method_2_1(self, n_splits=5):
        fold_dl = []
        for fold_idx in range(n_splits):
            fold_dict = {}
            # DataFrame
            ddi_val_df = self.ddi_label_df.query(
                f"affected_fold == {fold_idx}")
            train_fold_idx = set(range(n_splits)) - set([fold_idx])
            ddi_train_df = self.ddi_label_df[
                self.ddi_label_df['affected_fold'].isin(train_fold_idx)]

            # Concatenate the features
            if self.concat_features:
                ddi_train_df = pd.concat(
                [
                    ddi_train_df,
                    ddi_train_df.rename(
                        columns={"Subject_DrugbankID": "Affected_DrugbankID", 
                                "Affected_DrugbankID": "Subject_DrugbankID"})
                ],
                    axis= 0
                )
                ddi_val_df = pd.concat(
                [
                    ddi_val_df,
                    ddi_val_df.rename(
                        columns={"Subject_DrugbankID": "Affected_DrugbankID", 
                                "Affected_DrugbankID": "Subject_DrugbankID"})
                ],      
                    axis= 0
                )
            ddi_train_dt = self.DDIDataSet(ddi_train_df, self.features_df)
            ddi_val_dt = self.DDIDataSet(ddi_val_df, self.features_df)

            # DataLoader
            fold_dict['val'] = DataLoader(
                ddi_val_dt,
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=self.num_workers)

            fold_dict['train'] = DataLoader(
                ddi_train_dt,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            fold_dl.append(fold_dict)
        
        return fold_dl

    def _split_method_3(self, n_splits=5):
        fold_dl = []
        for fold_idx in range(n_splits):
            test_fold_bool = (self.ddi_label_df['subject_fold'] == fold_idx) | (self.ddi_label_df['affected_fold'] == fold_idx)
            ddi_val_df = self.ddi_label_df[test_fold_bool]
            ddi_val_df = ddi_val_df[
                ddi_val_df.subject_fold != ddi_val_df.affected_fold
            ]
            ddi_train_df = self.ddi_label_df[~test_fold_bool]
            fold_dict = {}
            # Concatenate the features
            if self.concat_features:
                ddi_train_df = pd.concat(
                [
                    ddi_train_df,
                    ddi_train_df.rename(
                        columns={"Subject_DrugbankID": "Affected_DrugbankID", 
                                "Affected_DrugbankID": "Subject_DrugbankID"})
                ],
                    axis= 0
                )
                ddi_val_df = pd.concat(
                [
                    ddi_val_df,
                    ddi_val_df.rename(
                        columns={"Subject_DrugbankID": "Affected_DrugbankID", 
                                "Affected_DrugbankID": "Subject_DrugbankID"})
                ],      
                    axis= 0
                )
            ddi_train_dt = self.DDIDataSet(ddi_train_df, self.features_df)
            ddi_val_dt = self.DDIDataSet(ddi_val_df, self.features_df)

            # DataLoader
            fold_dict['val'] = DataLoader(
                ddi_val_dt,
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=self.num_workers)

            fold_dict['train'] = DataLoader(
                ddi_train_dt,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            fold_dl.append(fold_dict)
        
        return fold_dl

    def _split_method_4(self, n_splits=5):
        fold_dl = []
        for fold_idx in range(n_splits):
            test_fold_bool = (self.ddi_label_df['subject_fold'] == fold_idx) & (self.ddi_label_df['affected_fold'] == fold_idx)
            list_folds = list(range(n_splits))
            list_folds.pop(fold_idx)
            train_fold_bool = (self.ddi_label_df['subject_fold'].isin(list_folds)) & (self.ddi_label_df['affected_fold'].isin(list_folds))
            ddi_val_df = self.ddi_label_df[test_fold_bool]
            ddi_train_df = self.ddi_label_df[train_fold_bool]
            fold_dict = {}
            # Concatenate the features
            if self.concat_features:
                ddi_train_df = pd.concat(
                [
                    ddi_train_df,
                    ddi_train_df.rename(
                        columns={"Subject_DrugbankID": "Affected_DrugbankID", 
                                "Affected_DrugbankID": "Subject_DrugbankID"})
                ],
                    axis= 0
                )
                ddi_val_df = pd.concat(
                [
                    ddi_val_df,
                    ddi_val_df.rename(
                        columns={"Subject_DrugbankID": "Affected_DrugbankID", 
                                "Affected_DrugbankID": "Subject_DrugbankID"})
                ],      
                    axis= 0
                )
            ddi_train_dt = self.DDIDataSet(ddi_train_df, self.features_df)
            ddi_val_dt = self.DDIDataSet(ddi_val_df, self.features_df)

            # DataLoader
            fold_dict['val'] = DataLoader(
                ddi_val_dt,
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=self.num_workers)

            fold_dict['train'] = DataLoader(
                ddi_train_dt,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            fold_dl.append(fold_dict)
        
        return fold_dl

    def _split_method_5(self, n_splits=5):
        fold_dl = []
        for fold_idx in range(n_splits):
            train_fold_bool = (self.ddi_label_df['subject_fold'] == fold_idx) ^ (self.ddi_label_df['affected_fold'] == fold_idx)
            ddi_train_df = self.ddi_label_df[train_fold_bool]
            test_fold_bool = (self.ddi_label_df['subject_fold'] == fold_idx) & (self.ddi_label_df['affected_fold'] == fold_idx)
            ddi_val_df = self.ddi_label_df[test_fold_bool]
            fold_dict = {}
            # Concatenate the features
            if self.concat_features:
                ddi_train_df = pd.concat(
                [
                    ddi_train_df,
                    ddi_train_df.rename(
                        columns={"Subject_DrugbankID": "Affected_DrugbankID", 
                                "Affected_DrugbankID": "Subject_DrugbankID"})
                ],
                    axis= 0
                )
                ddi_val_df = pd.concat(
                [
                    ddi_val_df,
                    ddi_val_df.rename(
                        columns={"Subject_DrugbankID": "Affected_DrugbankID", 
                                "Affected_DrugbankID": "Subject_DrugbankID"})
                ],      
                    axis= 0
                )
            ddi_train_dt = self.DDIDataSet(ddi_train_df, self.features_df)
            ddi_val_dt = self.DDIDataSet(ddi_val_df, self.features_df)

            # DataLoader
            fold_dict['val'] = DataLoader(
                ddi_val_dt,
                batch_size=self.batch_size, 
                shuffle=False,
                num_workers=self.num_workers)

            fold_dict['train'] = DataLoader(
                ddi_train_dt,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            fold_dl.append(fold_dict)
        
        return fold_dl
    def split(self, method, n_splits=5):
        if method == 'v1':
            return self._split_method_1(n_splits)
        elif method == 'v2':
            return self._split_method_2(n_splits)
        elif method == 'v2_1':
            return self._split_method_2_1(n_splits)
        elif method == 'v3':
            return self._split_method_3(n_splits)
        elif method == 'v4':
            return self._split_method_4(n_splits)
        elif method == 'v5':
            return self._split_method_5(n_splits)
        else:
            raise ValueError("Invalid splitting method. Choose from 1, 2, 3, or 4.")

# Example usage:
# splitter = DataSplitterByLevel(ddi_label_df, features_df, DDIDataSet)
# fold_dl = splitter.split(method='v1')  # Choose method 1, 2, 3, or 4