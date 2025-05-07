 #%%
from dataclasses import dataclass
import argparse
import sys
from pathlib import Path
# %%
# ----------------- #
def parse_folds(value):
    parts = [part.strip() for part in value.split(',') if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Invalid input for folds")
    if len(parts) == 1 and ',' not in value:
        return int(parts[0])
    return [int(part) for part in parts]


def get_args_MLP():
    """Get command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train MLP model on a dataset')
    splitter_list= ["RandomSplitter",
                    "ScaffoldSplitter",
                    "MolecularWeightSplitter",
                    "MaxMinSplitter",
                    "ButinaSplitter", 
                    "FingerprintSplitter"]
    parser.add_argument('--splitter', type=str, 
                        choices=splitter_list, 
                        default="ScaffoldSplitter",
                        help='Type of splitter to use. Default is RandomSplitter.')

    splitter_list = ["v1", "v2", "v2_1", "v3", "v4", "v5"]
    parser.add_argument('--split_level', type=str,
                        choices=splitter_list, default='v1')
    parser.add_argument('--classification_type', type=str,
                        choices=["multi-label", "multi-class"],
                        default="multi-class",
                        help='Type of classification to use. Default is multi-class.')
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="Weight decay (default: 0)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs (default: 1)")
    parser.add_argument("--folds", type=parse_folds, default=5,
                        help="Number of folds for cross-validation.\
                            Use a single integer or comma-separated integers\
                                (default: 1)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--dropout_rate", type=float, default=0.35,
                        help="Dropout rate (default: 0.3)")
    parser.add_argument("--enable_early_stopping", action="store_true",
                        help="Enable early stopping")
    parser.add_argument("--concat_features", action="store_true",
                        help="Concatenate features of subject and affected drugs")
    parser.add_argument("--intermediate_layer_num_severity", type=int, default=1,
                        help="The number of intermediate layer for severity (default: 1)")
    parser.add_argument("--intermediate_layer_num_desc", type=int, default=1,
                        help="The number of intermediate layer for desc(default: 1)")
    parser.add_argument("--num_workers_for_dt", default=4, type=int,
                        help="num workers for pytorch dataloader")
    if 'ipykernel' in sys.modules:
        return parser.parse_args([])
    else:
        return parser.parse_args()

def get_args_GCN():
    parser = argparse.ArgumentParser(
        description='Train GNN model on a dataset')
    splitter_list = ["v1", "v2", "v3", "v4"]
    parser.add_argument('--split_level', type=str,
                        choices=splitter_list, default='v1')
    parser.add_argument('--classification_type', type=str,
                        choices=["multi-label", "multi-class"],
                        default="multi-class",
                        help='Type of classification to use. Default is multi-class.')
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay (default: 0.0001)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs (default: 1)")
    parser.add_argument("--folds", type=parse_folds, default=1,
                        help="Number of folds for cross-validation.\
                            Use a single integer or comma-separated integers\
                                (default: 1)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size (default: 512)")
    parser.add_argument("--dropout_rate", type=float, default=0.35,
                        help="Dropout rate (default: 0.3)")
    parser.add_argument("--enable_early_stopping", action="store_true",
                        help="Enable early stopping")
    parser.add_argument("--expand_factor", default=2, type=int,
                        help="Expand factor for the graph convolutional layers")
    parser.add_argument("--num_workers_for_dt", default=4, type=int,
                        help="num workers for pytorch dataloader")
    if 'ipykernel' in sys.modules:
        return parser.parse_args([])
    else:
        return parser.parse_args()

def get_args_Transformer():
    parser = argparse.ArgumentParser(
        description='Train model on a dataset')
    splitter_list = ["v1", "v2", "v3", "v4"]
    parser.add_argument('--split_level', type=str,
                        choices=splitter_list, default='v1')
    parser.add_argument('--classification_type', type=str,
                        choices=["multi-label", "multi-class"],
                        default="multi-class",
                        help='Type of classification to use. Default is multi-class.')
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay (default: 0.0001)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs (default: 1)")
    parser.add_argument("--folds", type=parse_folds, default=1,
                        help="Number of folds for cross-validation.\
                            Use a single integer or comma-separated integers\
                                (default: 1)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size (default: 512)")
    parser.add_argument("--dropout_rate", type=float, default=0.35,
                        help="Dropout rate (default: 0.3)")
    parser.add_argument("--enable_early_stopping", action="store_true",
                        help="Enable early stopping")
    parser.add_argument("--expand_factor", default=2, type=int,
                        help="Expand factor for the graph convolutional layers")

    parser.add_argument("--num_workers_for_dt", default=4, type=int,
                        help="num workers for pytorch dataloader")
    parser.add_argument("--nhead", type=int, default=2,
        help="The number of heads in the multiheadattention models")    
    parser.add_argument("--num_encoder_layers", type=int, default=4,
        help="The number of sub-encoder-layers in the encoder")

    if 'ipykernel' in sys.modules:
        return parser.parse_args([])
    else:
        return parser.parse_args()
@dataclass(frozen=True)
class FilePaths:
    base_dir: Path = Path("./data")
    # Define file paths relative to the base directory
    ddi_label: Path = base_dir / "df_DDI_info.parquet"
    smiles: Path = base_dir / "DrugbankToPubchem_CanonicalSmiles_crawling_13117drugs.csv"
    pubchem_mapping: Path = base_dir / "df_pubchemID_mapped.parquet"

    # Method to check if the file paths are valid
    def check_paths_exist(self):
        for attribute, path in self.__dict__.items():
            if not path.exists():
                raise FileNotFoundError(f"Path '{attribute}' does not exist: {path}")
            else:
                print(f"Path '{attribute}' exists: {path}")
    def as_dict(self):
        return {
            'ddi_label': self.ddi_label,
            'smiles': self.smiles,
            'pubchem_mapping': self.pubchem_mapping
        }

# usage
# file_paths = FilePaths()
# file_paths.check_paths_exist()

from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class BaseFilePaths(ABC):
    """Base class for file paths with common functionality"""
    base_dir: Path = Path("/data/NoahsARK/DDI")

    @property
    @abstractmethod
    def ddi_label(self) -> Path:
        """Abstract property that concrete classes must implement"""
        pass

    @property
    def features(self) -> Path:
        return self.base_dir / "DDI_versionUp_for_paper/Split/scaffold_split_method/processed_data/features_ssp_admet_sumdti_concatenated_hIndex.parquet"

    @property
    def smiles(self) -> Path:
        return self.base_dir / "processed_data/DrugbankToPubchem_CanonicalSmiles_crawling_13117drugs.csv"

    @property
    def pubchem_mapping(self) -> Path:
        return self.base_dir / "DDI_versionUp_for_paper/0_Model_experiments/data/df_pubchemID_mapped.parquet"

    def check_paths_exist(self):
        """Check if all defined paths exist"""
        properties = [attr for attr in dir(self) if isinstance(getattr(type(self), attr, None), property)]
        for prop in properties:
            path = getattr(self, prop)
            if isinstance(path, Path):
                if not path.exists():
                    raise FileNotFoundError(f"Path '{prop}' does not exist: {path}")
                print(f"Path '{prop}' exists: {path}")

    def as_dict(self):
        """Return a dictionary of main file paths"""
        return {
            'ddi_label': self.ddi_label,
            'features': self.features,
            'smiles': self.smiles,
            'pubchem_mapping': self.pubchem_mapping
        }
