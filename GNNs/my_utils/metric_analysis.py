# metric_analyzer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
import re
import glob

import torchmetrics

class MetricAnalyzer:
    def __init__(self, log_path, plots_result=True):
        self.log_path = log_path
        self.output_dir = Path(log_path)
        self.plots_dir = self.output_dir / 'plots'
        self.tables_dir = self.output_dir / 'tables'
        self.plots_result = plots_result
        self.metric_groups = {
            'loss': {
                'metrics': ['train_total_loss', 'val_total_loss'],
                'title': 'Training and Validation Total Loss',
                'ylabel': 'Loss'
            },
            'f1_desc': {
                'metrics': ['train_f1_macro_desc', 'train_f1_micro_desc', 'train_f1_weighted_desc',
                           'val_f1_macro_desc', 'val_f1_micro_desc', 'val_f1_weighted_desc'],
                'title': 'F1 Scores for Description',
                'ylabel': 'F1 Score'
            },
            'f1_severity': {
                'metrics': ['train_f1_macro_severity', 'train_f1_micro_severity', 'train_f1_weighted_severity',
                           'val_f1_macro_severity', 'val_f1_micro_severity', 'val_f1_weighted_severity'],
                'title': 'F1 Scores for Severity',
                'ylabel': 'F1 Score'
            }
        }
        
    def setup_directories(self):
        """Create necessary output directories"""
        for dir_path in [self.output_dir, self.plots_dir, self.tables_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def plot_metric(self, df, metrics, title, ylabel, fold, output_path):
        """Plot metrics and save to file"""
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            sns.lineplot(x='epoch', y=metric, data=df, label=metric)
        plt.title(f'{title} (Fold {fold})')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def load_metric_files(self):
        """Load and process all metric files"""
        metrics_files = glob.glob(os.path.join(self.log_path, "**/metrics.csv"), recursive=True)
        fold_dfs = {}
        
        for file in metrics_files:
            version_match = re.search(r'version_(\d+)', file)
            if version_match:
                fold_num = version_match.group(1)
                df = pd.read_csv(file)
                df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
                df['fold'] = f'fold_{fold_num}'
                fold_dfs[f'fold_{fold_num}'] = df
                
        return fold_dfs
    
    def process_summaries(self, fold_dfs):
        """Process and combine fold summaries"""
        combined_df = pd.concat(fold_dfs.values(), ignore_index=True)
        metrics = [col for col in combined_df.columns if col not in ['epoch', 'fold']]
        
        fold_summaries = []
        for fold, df in fold_dfs.items():
            fold_summary = df.groupby('epoch')[metrics].mean()
            fold_summary['fold'] = fold
            fold_summaries.append(fold_summary)
            
        all_folds_summary = pd.concat(fold_summaries)
        all_folds_summary = all_folds_summary.drop(columns='step', axis=1)
        all_folds_summary = all_folds_summary.sort_values(
            'fold', 
            key=lambda x: x.str.extract('(\d+)')[0].astype(int)
        )
        
        return all_folds_summary
    
    def create_best_metrics_summary(self, all_folds_summary):
        """Create summary of best metrics"""
        best_df = all_folds_summary.reset_index()
        best_df = best_df.loc[best_df.groupby('fold')['val_f1_macro_desc'].idxmax()]
        
        means = best_df.select_dtypes(include=['float', 'int']).mean()
        mean_row = pd.DataFrame([{**means, 'fold': 'average'}])
        best_df = pd.concat([best_df, mean_row], ignore_index=True)
        
        fold_col = best_df.pop('fold')
        best_df.insert(0, 'fold', fold_col)
        
        return best_df
    
    def generate_plots(self, fold_dfs):
        """Generate plots for each fold and metric group"""
        for fold, df in fold_dfs.items():
            fold_dir = self.plots_dir / fold
            fold_dir.mkdir(exist_ok=True)
            
            df_grouped = df.drop('fold', axis=1).groupby('epoch').mean().reset_index()
            for group_name, group_info in self.metric_groups.items():
                output_path = fold_dir / f'{group_name}_{fold}.png'
                self.plot_metric(
                    df_grouped,
                    group_info['metrics'],
                    group_info['title'],
                    group_info['ylabel'],
                    fold,
                    output_path
                )
    
    def run_analysis(self):
        """Run the complete analysis pipeline"""
        # Setup directories
        self.setup_directories()
        
        # Load and process data
        fold_dfs = self.load_metric_files()
        all_folds_summary = self.process_summaries(fold_dfs)
        all_folds_summary.to_csv(self.tables_dir / 'all_folds_summary.csv', index=True)
        
        # Create best metrics summary
        best_df = self.create_best_metrics_summary(all_folds_summary)
        best_df.to_csv(self.tables_dir / 'best_metric_5_fold.csv', index=False)
        
        # Generate plots
        if self.plots_result:
            self.generate_plots(fold_dfs)
        
        return best_df

def analyze_metrics(log_path, plots_result=True):
    """Main function to analyze metrics"""
    analyzer = MetricAnalyzer(log_path, plots_result)
    return analyzer.run_analysis()



# -----------------
import torch
import re
from typing import List, Dict, Tuple, Any
import lightning as L

class MLPEvaluatorAverageBidirection:
    def __init__(self, 
                 model_class,
                 lit_model_class,
                 input_shape: int,
                 hidden_size: int,
                 intermediate_layers: int,
                 output_shapes: Dict,
                 output_configs: Dict,
                 ddi_processor: Any,
                 device: int=0):
                 
        """
        Initialize the model evaluator.
        """
        self.model_class = model_class
        self.lit_model_class = lit_model_class
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.intermediate_layers = intermediate_layers
        self.output_shapes = output_shapes
        self.output_configs = output_configs
        self.ddi_processor = ddi_processor
        self.metric_desc = torchmetrics.classification.MulticlassF1Score(
            num_classes=self.ddi_processor.NUM_DESC_CLASSES, average=None)
        self.metric_severity = torchmetrics.classification.MulticlassF1Score(
            num_classes=self.ddi_processor.NUM_SEVERITY_CLASSES, average=None)
        self.device = device
        self.trainer = L.Trainer(devices=[self.device])

    def _extract_fold_number(self, model_ckpt: str) -> int:
        """Extract fold number from checkpoint path."""
        return int(re.search(r'fold_(\d+)_best_model', str(model_ckpt)).group(1))

    def _process_labels(self, val_loader) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process and concatenate labels from dataloader."""
        severity_list = []
        desc_list = []
        
        for batch in val_loader:
            severity_labels, desc_labels = batch[1], batch[2]
            severity_list.append(severity_labels)
            desc_list.append(desc_labels)
            
        severity_labels = torch.cat(severity_list, dim=0)
        desc_labels = torch.cat(desc_list, dim=0)
        
        severity_labels = severity_labels.reshape(2, -1, self.ddi_processor.NUM_SEVERITY_CLASSES)[0]
        desc_labels = desc_labels.reshape(2, -1, self.ddi_processor.NUM_DESC_CLASSES)[0]
            
        return severity_labels, desc_labels

    def _load_model(self, model_ckpt: str, args) -> Any:
        """Load model from checkpoint."""
        model = self.model_class(
            self.input_shape,
            self.hidden_size,
            self.intermediate_layers,
            self.output_shapes,
            dropout_rate=args.dropout_rate
        )
        
        lit_model = self.lit_model_class.load_from_checkpoint(
            model_ckpt,
            model=model,
            output_configs=self.output_configs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        return lit_model

    def _process_predictions(self, preds: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process model predictions."""
        preds_severity_prob_list = [pred[0]['probabilities'] for pred in preds]
        preds_desc_prob_list = [pred[1]['probabilities'] for pred in preds]
        
        preds_severity_t = torch.cat(preds_severity_prob_list, dim=0)
        preds_desc_t = torch.cat(preds_desc_prob_list, dim=0)
        
        preds_severity_t = preds_severity_t.reshape(
            2, -1, self.ddi_processor.NUM_SEVERITY_CLASSES)
        preds_desc_t = preds_desc_t.reshape(
            2, -1, self.ddi_processor.NUM_DESC_CLASSES)
        preds_severity_t = preds_severity_t.mean(dim=0)
        preds_desc_t = preds_desc_t.mean(dim=0)
            
        return preds_severity_t, preds_desc_t

    def evaluate_fold(self, model_ckpt: str, fold_dl: Dict, args) -> Dict[str, float]:
        """Evaluate a single fold."""
        fold_num = self._extract_fold_number(model_ckpt)
        val_loader = fold_dl[fold_num]['val']
        
        print(f"Evaluating Fold {fold_num}")
        
        # Process labels
        severity_labels, desc_labels = self._process_labels(val_loader)
        
        # Load and predict
        lit_model = self._load_model(model_ckpt, args)
        preds = self.trainer.predict(lit_model, val_loader)
        
        # Process predictions
        preds_severity_t, preds_desc_t = self._process_predictions(preds)
        
        # Calculate metrics
        severity_f1 = self.metric_severity(
            preds_severity_t, 
            torch.argmax(severity_labels, dim=1)
        ).mean()
        
        desc_f1 = self.metric_desc(
            preds_desc_t, 
            torch.argmax(desc_labels, dim=1)
        ).mean()
        
        return {
            'fold': fold_num,
            'severity_f1': severity_f1.item(),
            'desc_f1': desc_f1.item()
        }

    def evaluate_all_folds(self, model_ckpt_lst: List[str], fold_dl: Dict, args) -> List[Dict[str, float]]:
        """Evaluate all folds."""
        results = []
        for model_ckpt in model_ckpt_lst:
            fold_results = self.evaluate_fold(model_ckpt, fold_dl, args)
            results.append(fold_results)
            print(f"Fold {fold_results['fold']}: "
                  f"Severity F1 = {fold_results['severity_f1']:.4f}, "
                  f"Desc F1 = {fold_results['desc_f1']:.4f}")
        return results

# Usage example:
"""
evaluator = ModelEvaluator(
    model_class=MLPDynamicMultiOutputModel,
    lit_model_class=MLPFlexibleDynamicLitMultiOutputModel,
    input_shape=input_shape,
    hidden_size=hidden_size,
    intermediate_layers=intermediate_layers,
    output_shapes=output_shapes,
    output_configs=output_configs,
    ddi_processor=ddi_processor,
    metric_severity=metric_severity,
    metric_desc=metric_desc,
    mean_tag=True
)

results = evaluator.evaluate_all_folds(model_ckpt_lst, fold_dl, args)
"""