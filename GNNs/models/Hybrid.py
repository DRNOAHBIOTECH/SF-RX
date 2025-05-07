import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import torchmetrics

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import GCNConv

class HybridGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 input_mlp_shape, hidden_mlp_shape, hidden_mlp_size,
                 output_configs, dropout_rate):
        super(HybridGCN, self).__init__()

        # GCN part
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.output_configs = output_configs

        # MLP part
        self.norm_mlp = nn.BatchNorm1d(input_mlp_shape)
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(input_mlp_shape if i == 0 else hidden_mlp_shape, hidden_mlp_shape),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_mlp_shape),
                    nn.Dropout(dropout_rate)
                ) for i in range(hidden_mlp_size)
            ]
        )

        # The varying number of Multiple output layers
        self.output_layers = nn.ModuleList([
            torch.nn.Linear(hidden_channels * 3 + hidden_mlp_shape, config['num_classes'])
            for config in self.output_configs
        ])

        self.dropout_rate = dropout_rate
    
    def forward(self, batch):
        mlp_x = batch[0]
        gcn_x = batch[1].x
        gcn_edge_index = batch[1].edge_index
        gcn_batch = batch[1].batch

        # 1. Obtain node embeddings
        gcn_x = self.conv1(gcn_x, gcn_edge_index)
        gcn_x = self.bn1(gcn_x)
        gcn_x = F.relu(gcn_x) 
        # x = F.leaky_relu(x)
        prev_x = gcn_x

        gcn_x = self.conv2(gcn_x, gcn_edge_index)
        gcn_x = self.bn2(gcn_x)
        gcn_x = gcn_x + prev_x
        gcn_x = F.relu(gcn_x)
        # x = F.leaky_relu(x)
        prev_x = gcn_x

        gcn_x = self.conv3(gcn_x, gcn_edge_index)
        gcn_x = F.relu(gcn_x)
        # x = F.leaky_relu(x)
        # x = F.tanh(x)

        # 2. Readout layer
        # Global pooling(stack different aggregations)
        gcn_x = torch.cat([global_mean_pool(gcn_x, gcn_batch),
                           global_add_pool(gcn_x, gcn_batch),
                           global_max_pool(gcn_x, gcn_batch)], dim=1)
        # 3. Apply a final classifier
        gcn_x= F.dropout(gcn_x, p=self.dropout_rate, training=self.training)
        
        # MLP part
        mlp_x = self.norm_mlp(mlp_x)
        mlp_x = self.hidden_layers(mlp_x)

        # Concatenate the outputs of the GCN and MLP
        x = torch.cat([gcn_x, mlp_x], dim=1)

        # Apply multiple output layers
        outputs = []
        for layer, config in zip(self.output_layers, self.output_configs):
            out = layer(x)
            # if config.get('activation') == 'sigmoid':
            #     out = torch.sigmoid(out)
            outputs.append(out)
        return outputs

class FlexibleLitHybridModel(L.LightningModule):
    def __init__(self, model, 
                output_configs, 
                learning_rate, 
                weight_decay, 
                batch_size):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_configs = output_configs
        self.batch_size = batch_size
        self.train_metrics = nn.ModuleList()
        self.val_metrics = nn.ModuleList()
        self.loss_fns = nn.ModuleList()
        self.loss_weights = []
        
        for config in self.output_configs:
            if config['classification_type'] == 'multi-label':
                loss_fn = nn.BCEWithLogitsLoss()
                f1_metrics = {
                    'macro': torchmetrics.classification.MultilabelF1Score(num_labels=config['num_classes'], average='macro'),
                    'micro': torchmetrics.classification.MultilabelF1Score(num_labels=config['num_classes'], average='micro'),
                    'weighted': torchmetrics.classification.MultilabelF1Score(num_labels=config['num_classes'], average='weighted')
                }
            else:  # multi-class (works with one-hot encoded targets)
                loss_fn = nn.CrossEntropyLoss()
                f1_metrics = {
                    'macro': torchmetrics.classification.MulticlassF1Score(num_classes=config['num_classes'], average='macro'),
                    'micro': torchmetrics.classification.MulticlassF1Score(num_classes=config['num_classes'], average='micro'),
                    'weighted': torchmetrics.classification.MulticlassF1Score(num_classes=config['num_classes'], average='weighted')
                }
            
            self.loss_fns.append(loss_fn)
            self.train_metrics.append(nn.ModuleDict({
                **{f'f1_{avg}': metric.clone() for avg, metric in f1_metrics.items()}
            }))
            self.val_metrics.append(nn.ModuleDict({
                **{f'f1_{avg}': metric.clone() for avg, metric in f1_metrics.items()}
            }))
            self.loss_weights.append(config['loss_weight'])

    def _shared_step(self, batch, batch_idx, metrics, stage):
        outputs = self.model(batch)
        total_loss = 0
        for i, (pred, config) in enumerate(zip(outputs, self.output_configs)):
            target = batch[1][config['target']]
            if config['classification_type'] == 'multi-class':
                # For multi-class, target can be one-hot or class indices
                if target.dim() == 2:  # one-hot encoded
                    target_indices = torch.argmax(target, dim=1)
                else:  # already class indices
                    target_indices = target
            else:  # multi-label
                target = target.float()
                target_indices = target

            loss = self.loss_fns[i](pred, target)
            total_loss += self.loss_weights[i] * loss
            
            self.log(f'{stage}_loss_{config["name"]}', loss, prog_bar=True,
                     batch_size=self.batch_size)
            
            if config['classification_type'] == 'multi-class':
                pred_probs = torch.softmax(pred, dim=1)
                pred_indices = torch.argmax(pred_probs, dim=1)
            else:  # multi-label
                pred_probs = torch.sigmoid(pred)
                pred_indices = (pred_probs > 0.5).float()
            
            for avg in ['macro', 'micro', 'weighted']:
                metrics[i][f'f1_{avg}'](pred_indices, target_indices)
                self.log(f'{stage}_f1_{avg}_{config["name"]}', 
                         metrics[i][f'f1_{avg}'], 
                         prog_bar=True, on_step=False, on_epoch=True)
        
        self.log(f'{stage}_total_loss', total_loss, prog_bar=True, 
                 batch_size=self.batch_size)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.train_metrics, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.val_metrics, 'val')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self.model(batch.x.float(), batch.edge_index, batch.batch)
        predictions = []
        for i, (pred, config) in enumerate(zip(outputs, self.output_configs)):
            pred_probs = torch.sigmoid(pred)
            if config['classification_type'] == 'multi-class':
                pred_labels = torch.argmax(pred_probs, dim=1)
            else:  # multi-label
                pred_labels = (pred_probs > 0.5).float()
            
            predictions.append({
                'task_name': config['name'],
                'probabilities': pred_probs,
                'predictions': pred_labels
            })
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.learning_rate, 
                                      weight_decay=self.weight_decay)
        return optimizer
