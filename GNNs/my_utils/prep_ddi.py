import pandas as pd
from pathlib import Path
from configparser import ConfigParser
from sqlalchemy import create_engine
from typing import Dict, Set, Optional

import numpy as np
from itertools import chain
from sklearn.model_selection import StratifiedKFold


class DDIDataPreparation:
    def __init__(self, file_paths: Dict[str, str]) -> None:
        self.file_paths: Dict[str, Path] = {k: Path(v) for k, v in file_paths.items()}
        self.ddi_label_df: Optional[pd.DataFrame] = None
        self.smiles_df: Optional[pd.DataFrame] = None
        self.pubchemID_mapped_df: Optional[pd.DataFrame] = None
        self.dbid_info_df: Optional[pd.DataFrame] = None
        self.db_id_to_smiles_dict: Optional[Dict[str, str]] = None

    def _load_config(self, config_path: str) -> ConfigParser:
        config = ConfigParser()
        config.read(config_path)
        return config

    def load_ddi_label_data(self) -> None:
        self.ddi_label_df = pd.read_parquet(self.file_paths['ddi_label'])

    def load_smiles_data(self) -> None:
        self.smiles_df = pd.read_csv(self.file_paths['smiles'])
        self.smiles_df.loc[:, "PubChemCID"] = (
            self.smiles_df.PubChemCID.str.split(":").apply(lambda x: x[1])
        )
        self.smiles_df = self.smiles_df.loc[~self.smiles_df.drugbank_id.str.startswith("DBSALT")]
        self.db_id_to_smiles_dict = dict(zip(self.smiles_df.drugbank_id, self.smiles_df.smiles))

    def load_pubchemID_mapping(self) -> None:
        self.pubchemID_mapped_df = pd.read_parquet(self.file_paths['pubchem_mapping'])
        self._check_pubchemID_mapping()

    def _check_pubchemID_mapping(self) -> None:
        pubchemID_is_unique: bool = self.pubchemID_mapped_df["pubchem_ids"].is_unique
        drugbankID_is_unique: bool = self.pubchemID_mapped_df["dbid"].is_unique
        print(f'pubchem ID is unique: {pubchemID_is_unique}')
        print(f'Drugbank ID is unique: {drugbankID_is_unique}')
        if not pubchemID_is_unique and drugbankID_is_unique:
            print("The mapping table is not one-to-one.\n"
                  "One drugbankID is mapped to multiple PubChemIDs.")

    def filter_data(self) -> None:
        print(f'Number of drug-drug interactions before filtering: {self.ddi_label_df.shape[0]}')
        
        # 3. Select drug-drug interactions with available SMILES strings
        self.ddi_label_df = self.ddi_label_df.query(
            "Subject_DrugbankID in @self.smiles_df.drugbank_id and Affected_DrugbankID in @self.smiles_df.drugbank_id"
        )
        print(f'Number of drug-drug interactions after filtering 3(with available SMILES strings): {self.ddi_label_df.shape[0]}')
        
        # Remove duplicated drug drug interactions
        self.ddi_label_df = self.ddi_label_df.drop_duplicates(subset=["Subject_DrugbankID", "Affected_DrugbankID"])
        print(f'Number of drug-drug interactions after removing duplicate DDI: {self.ddi_label_df.shape[0]}')

    def prepare_data(self) -> None:
        self.load_ddi_label_data()
        self.load_smiles_data()
        self.load_pubchemID_mapping()
        self.filter_data()

    def get_prepared_data(self) -> Dict[str, Optional[pd.DataFrame] | Optional[Dict[str, str]] | Optional[Set[str]]]:
        return {
            'ddi_label_df': self.ddi_label_df,
            'smiles_df': self.smiles_df,
            'pubchemID_mapped_df': self.pubchemID_mapped_df,
            'db_id_to_smiles_dict': self.db_id_to_smiles_dict,
        }


def filter_categorical_by_count(df: pd.DataFrame, column: str, min_count: int) -> pd.DataFrame:
    """
    Filter a DataFrame to keep only rows where the categorical variable
    in the specified column appears at least min_count times.

    :param df: Input DataFrame
    :param column: Name of the column containing categorical variables
    :param min_count: Minimum count threshold for categories to keep
    :return: Filtered DataFrame
    """
    # Get value counts
    value_counts = df[column].value_counts()

    # Create a boolean mask for categories meeting the threshold
    mask = df[column].isin(value_counts[value_counts > min_count].index)

    # Apply the mask to filter the DataFrame
    return df[mask]

# ddi_data_preparation.py


# from config import get_args, FilePaths 
# from my_utils.prep_ddi import DDIDataPreparation, filter_categorical_by_count
# from dce_prepare_smiles import smiles_to_dce_features

class DDIDataProcessor:
    def __init__(self, file_paths, 
                no_desc_classes_cut_off=0,
                no_desc_classes_within_folds=10,
                class_distribution_method='v2'):
        self.file_paths = file_paths
        self.no_desc_classes_cut_off = no_desc_classes_cut_off 
        self.no_desc_classes_within_folds = no_desc_classes_within_folds
        self.class_distribution_method = class_distribution_method
        self.data_prep = None
        self.prepared_data = None
        self.ddi_label_df = None
        self.features_df = None
        self.smiles_df = None
        self.pubchemID_mapped_df = None
        self.db_id_to_smiles_dict = None
        self.db_id_set_for_ddi = None
        self.dict_desc_classes = None
        self.dict_severity_classes = None
        self.dict_action_classes = None
        self.NUM_DESC_CLASSES = None
        self.NUM_SEVERITY_CLASSES = None
        self.NUM_ACTION_CLASSES = None


    def prepare_data(self):
        self.data_prep = DDIDataPreparation(self.file_paths)
        self.data_prep.prepare_data()
        self.prepared_data = self.data_prep.get_prepared_data()
        
        self.ddi_label_df = self.prepared_data['ddi_label_df'].copy()
        self.smiles_df = self.prepared_data['smiles_df']
        self.pubchemID_mapped_df = self.prepared_data['pubchemID_mapped_df']
        self.db_id_to_smiles_dict = self.prepared_data['db_id_to_smiles_dict']
        
        self.db_id_set_for_ddi = (
            set(self.ddi_label_df['Subject_DrugbankID'].values) |
            set(self.ddi_label_df['Affected_DrugbankID'].values)
        )
        
        self.db_id_to_smiles_dict = dict(zip(self.smiles_df['drugbank_id'], self.smiles_df['smiles']))
        self.ddi_label_df = filter_categorical_by_count(
            self.ddi_label_df, 'type',
            self.no_desc_classes_cut_off)

    def create_class_mappings(self):
        self.dict_desc_classes = {desc:idx for idx, desc in enumerate(self.ddi_label_df.type.unique())}
        self.dict_severity_classes = {severity:idx for idx, severity in enumerate(self.ddi_label_df.Severity.unique())}
        
        self.NUM_DESC_CLASSES = len(self.dict_desc_classes)
        self.NUM_SEVERITY_CLASSES = len(self.dict_severity_classes)
        
        self.ddi_label_df.loc[:, 'desc_category_integer'] = self.ddi_label_df.loc[:, 'type'].map(self.dict_desc_classes)


    def distribute_classes(self):
        desc_class_per_two_way_fold = self.ddi_label_df.groupby(['subject_fold', 'affected_fold'])['desc_category_integer'].unique()
        count_desc_class_per_two_way_fold = desc_class_per_two_way_fold.explode().value_counts()
        
        all_included_classes = count_desc_class_per_two_way_fold[(count_desc_class_per_two_way_fold == 25)].index.to_list()
        
        all_included_df = self.ddi_label_df[self.ddi_label_df.desc_category_integer.isin(all_included_classes)].copy()
        not_all_included_df = self.ddi_label_df[~self.ddi_label_df.desc_category_integer.isin(all_included_classes)].copy()
        not_all_included_df = not_all_included_df.reset_index()
        
        fold_names = np.array([f"{i}_{j}" for i in range(5) for j in range(5)]).reshape(5, 5)
        fold_names = fold_names.ravel()
        skf = StratifiedKFold(n_splits=25, shuffle=True, random_state=42)
        not_included_folds = skf.split(not_all_included_df, not_all_included_df.desc_category_integer)
        not_included_folds = [ele[1] for ele in not_included_folds]
        not_included_folds_dict = dict(zip(fold_names, not_included_folds))
        
        for fold_name in not_included_folds_dict:
            subject_fold, affected_fold = fold_name.split("_")
            mask = not_all_included_df.index.isin(not_included_folds_dict[fold_name])
            not_all_included_df.loc[mask, "subject_fold"] = int(subject_fold) 
            not_all_included_df.loc[mask, "affected_fold"] = int(affected_fold) 
        
        self.ddi_label_df = pd.concat([all_included_df, not_all_included_df]).reset_index()
    
    def distribute_classesV2(self):
        ddi_label_df = self.ddi_label_df.copy()
        desc_class_count_per_two_way_fold = (
            ddi_label_df
            .groupby(['subject_fold', 'affected_fold'])['desc_category_integer']
            .value_counts()
        )
        desc_class_count_per_two_way_fold_df =(
            pd.DataFrame(desc_class_count_per_two_way_fold)
            .reset_index()
            .pivot(
                index=['subject_fold', 'affected_fold'],
                columns='desc_category_integer',
                values='desc_category_integer')
            .fillna(0)
        )

        all_included_classes =\
        (
            desc_class_count_per_two_way_fold_df
            .min(0)[
                desc_class_count_per_two_way_fold_df.min(0) >= self.no_desc_classes_within_folds
                ]
        ).index
        all_included_df = ddi_label_df[ddi_label_df.desc_category_integer.isin(all_included_classes)]
        self.ddi_label_df = all_included_df



    def process(self):
        self.prepare_data()
        self.create_class_mappings()
