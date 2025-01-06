from torch_geometric.nn import GCNConv, GATConv, GINConv 
from torch_geometric.nn import GATv2Conv, GlobalAttention
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torchmetrics
from torch_geometric.nn import LayerNorm, Linear, NNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, 
                 hidden_channels, 
                 output_configs, 
                 dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.output_configs = output_configs

        # The varying number of Multiple output layers
        self.output_layers = nn.ModuleList([
            torch.nn.Linear(hidden_channels * 3, config['num_classes'])
            for config in self.output_configs
        ])

        self.dropout_rate = dropout_rate
    
    def forward(self, data):
        # 1. Obtain node embeddings
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr,\
              data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x) 
        # x = F.leaky_relu(x)
        prev_x = x

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x + prev_x
        x = F.relu(x)
        # x = F.leaky_relu(x)
        prev_x = x

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x + prev_x
        x = F.relu(x)
        # x = F.leaky_relu(x)
        # x = F.tanh(x)

        # 2. Readout layer
        # Global pooling(stack different aggregations)
        x = torch.cat([global_mean_pool(x, batch), 
                       global_add_pool(x, batch), 
                       global_max_pool(x, batch)], dim=1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # 4. Apply multiple output layers
        outputs = []
        for layer, config in zip(self.output_layers, self.output_configs):
            out = layer(x)
            # if config.get('activation') == 'sigmoid':
            #     out = torch.sigmoid(out)
            outputs.append(out)
        return outputs


class FlexibleLitGNNModel(L.LightningModule):
    def __init__(self, 
                 model, 
                 output_configs, 
                 learning_rate, 
                 weight_decay, 
                 batch_size):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_configs = output_configs
        self.train_metrics = nn.ModuleList()
        self.val_metrics = nn.ModuleList()
        self.loss_fns = nn.ModuleList()
        self.loss_weights = []
        self.batch_size = batch_size
        
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
            target = batch[config['target']]
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
        outputs = self.model(batch)
        predictions = []
        for i, (pred, config) in enumerate(zip(outputs, self.output_configs)):
            if config['classification_type'] == 'multi-class':
                pred_probs = torch.softmax(pred, dim=1)
                pred_labels = torch.argmax(pred_probs, dim=1)
            else:  # multi-label
                pred_probs = torch.sigmoid(pred)
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

def generate_gnn_output_config(*class_configs):
    total_classes = sum(config['num_classes'] for config in class_configs)
    
    output_configs = []
    for config in class_configs:
        name = config['name']
        num_classes = config['num_classes']
        classification_type = config['classification_type']
        loss_weight = num_classes / total_classes
        
        output_configs.append({
            'name': name,
            'num_classes': num_classes,
            'loss_weight': loss_weight,
            'classification_type': classification_type,
            'target': f'{name}_y'
        })
    
    return output_configs

# Edge-conditioned Convolutional Layer
class NNConvLayer(nn.Module):
    def __init__(self, num_node_features, 
                 num_edge_features, 
                 output_configs,
                 dropout_rate, 
                 expand_factor):
        super().__init__()
        self.output_configs = output_configs
        conv1_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_node_features*num_node_features*expand_factor))
        conv2_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.Linear(32, num_node_features*expand_factor *num_node_features))
        self.conv1 = NNConv(num_node_features, num_node_features*expand_factor, conv1_net)
        self.conv2 = NNConv(num_node_features*expand_factor, num_node_features, conv2_net)
        self.fc_1 = nn.Linear(num_node_features, num_node_features*expand_factor)
        self.dropout = nn.Dropout(dropout_rate)
        # self.out = nn.Linear(num_node_features*expand_factor, NUM_DESC_CLASSES)

        # The varying number of Multiple output layers
        self.output_layers = nn.ModuleList([
            torch.nn.Linear(num_node_features*expand_factor, 
                            config['num_classes'])
            for config in self.output_configs
        ])
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index,\
              data.edge_attr, data.batch
        x = F.relu(self.conv1(x.float(), edge_index, edge_attr.float()))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr.float()))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc_1(x))
        x = self.dropout(x)
        # x = self.out(x)
        outputs = []
        for layer, config in zip(self.output_layers, self.output_configs):
            out = layer(x) # logits
            outputs.append(out)
        return outputs

# Edge-conditioned Convolutional Layer with improved capacity
# %%
class ECN(nn.Module):
    def __init__(self, num_node_features, 
                 num_edge_features, 
                 output_configs,
                 dropout_rate, 
                 expand_factor):
        super().__init__()
        self.output_configs = output_configs
        
        # Increased hidden dimension in edge networks for more capacity
        conv1_net = nn.Sequential(
            nn.Linear(num_edge_features, 64),  # Increased from 32 to 64
            nn.ReLU(),
            nn.Linear(64, num_node_features*num_node_features*expand_factor))
        
        # Middle layer maintains expanded dimension
        conv2_net = nn.Sequential(
            nn.Linear(num_edge_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_node_features*expand_factor*num_node_features*expand_factor))
            
        # Final convolution layer for contraction
        conv3_net = nn.Sequential(
            nn.Linear(num_edge_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_node_features*expand_factor*num_node_features))
        
        # Three NNConv layers with increasing then decreasing dimensions
        self.conv1 = NNConv(num_node_features, num_node_features*expand_factor, conv1_net)
        self.conv2 = NNConv(num_node_features*expand_factor, 
                           num_node_features*expand_factor, conv2_net)
        self.conv3 = NNConv(num_node_features*expand_factor, num_node_features, conv3_net)
        
        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm1d(num_node_features*expand_factor)
        self.bn2 = nn.BatchNorm1d(num_node_features*expand_factor)
        self.bn3 = nn.BatchNorm1d(num_node_features)
        
        # Modified: Adjusted FC layer input dimension for concatenated pooling
        self.fc_1 = nn.Linear(num_node_features * 3, num_node_features*expand_factor)  # *3 for concatenated pooling
        self.fc_2 = nn.Linear(num_node_features*expand_factor, num_node_features*expand_factor)
        
        # Lower dropout rate to avoid underfitting
        self.dropout = nn.Dropout(dropout_rate * 0.8)  # Reduced dropout rate
        
        # Output layers remain the same
        self.output_layers = nn.ModuleList([
            torch.nn.Linear(num_node_features*expand_factor, 
                          config['num_classes'])
            for config in self.output_configs
        ])
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index,\
              data.edge_attr, data.batch
        
        # First convolution with batch norm and residual connection
        identity = x
        x = self.conv1(x.float(), edge_index, edge_attr.float())
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second convolution with batch norm and residual connection
        prev_x = x
        x = self.conv2(x, edge_index, edge_attr.float())
        x = self.bn2(x)
        x = F.relu(x)
        x = x + prev_x  # Residual connection
        x = self.dropout(x)
        
        # Third convolution with batch norm
        x = self.conv3(x, edge_index, edge_attr.float())
        x = self.bn3(x)
        x = F.relu(x)
        
        # Modified: Multi-view pooling (concatenating mean, sum, and max pooling)
        x = torch.cat([
            global_mean_pool(x, batch),
            global_add_pool(x, batch),
            global_max_pool(x, batch)
        ], dim=1)
        
        # Additional fully connected layers with ReLU
        x = F.relu(self.fc_1(x))
        x = self.dropout(x)
        x = F.relu(self.fc_2(x))
        x = self.dropout(x)
        
        # Multiple output heads
        outputs = []
        for layer, config in zip(self.output_layers, self.output_configs):
            out = layer(x)
            outputs.append(out)
        
        return outputs


# GAT model with multiple outputs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool

class GAT(torch.nn.Module):
    def __init__(self, in_channels, 
                 hidden_channels, 
                 output_configs,
                 dropout_rate,
                 heads=8):  # Added heads parameter for multi-head attention
        super(GAT, self).__init__()
        
        # First GAT layer
        # Note: hidden_channels must be divisible by heads
        self.conv1 = GATConv(in_channels, 
                            hidden_channels // heads,
                            heads=heads,
                            dropout=dropout_rate)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Second GAT layer
        self.conv2 = GATConv(hidden_channels,
                            hidden_channels // heads,
                            heads=heads,
                            dropout=dropout_rate)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Third GAT layer
        self.conv3 = GATConv(hidden_channels,
                            hidden_channels // heads,
                            heads=heads,
                            dropout=dropout_rate)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        
        self.output_configs = output_configs
        # Multiple output layers
        self.output_layers = nn.ModuleList([
            torch.nn.Linear(hidden_channels * 3, config['num_classes'])
            for config in self.output_configs
        ])
        
        self.dropout_rate = dropout_rate
    
    def forward(self, data):
        # 1. Obtain node embeddings with attention
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # First attention layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)  # ELU is often used with GAT instead of ReLU
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        prev_x = x
        
        # Second attention layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x + prev_x  # Skip connection
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        prev_x = x
        
        # Third attention layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x + prev_x  # Skip connection
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # 2. Readout layer
        # Global pooling (stack different aggregations)
        x = torch.cat([global_mean_pool(x, batch),
                      global_add_pool(x, batch),
                      global_max_pool(x, batch)], dim=1)
        
        # 3. Apply multiple output layers
        outputs = []
        for layer, config in zip(self.output_layers, self.output_configs):
            out = layer(x)
            outputs.append(out)
            
        return outputs
class GATV2(torch.nn.Module):
    def __init__(self, in_channels, 
                 hidden_channels, 
                 output_configs,
                 dropout_rate,
                 heads=8,
                 attention_dropout=0.1):
        super().__init__()
        
        # Track dimensions for residual connections
        self.hidden_channels = hidden_channels
        
        # Projection layer for residual connection
        self.proj1 = nn.Linear(in_channels, hidden_channels * heads)
        
        # First GAT layer
        self.conv1 = GATConv(in_channels, 
                            hidden_channels,
                            heads=heads,
                            dropout=attention_dropout,
                            concat=True)  # Output will be hidden_channels * heads
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        # Second GAT layer
        self.conv2 = GATConv(hidden_channels * heads,
                            hidden_channels,
                            heads=heads,
                            dropout=attention_dropout,
                            concat=True)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        # Third GAT layer
        self.conv3 = GATConv(hidden_channels * heads,
                            hidden_channels,
                            heads=heads,
                            dropout=attention_dropout,
                            concat=True)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels * heads)
        
        # Feed-forward networks
        ff_dim = hidden_channels * heads
        self.ff1 = nn.Sequential(
            nn.Linear(ff_dim, ff_dim * 2),
            nn.LayerNorm(ff_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim * 2, ff_dim)  # Project back to residual dimension
        )
        
        self.ff2 = nn.Sequential(
            nn.Linear(ff_dim, ff_dim * 2),
            nn.LayerNorm(ff_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim * 2, ff_dim)  # Project back to residual dimension
        )
        
        # Layer scaling parameters
        self.layer_scale1 = nn.Parameter(torch.ones(1) * 0.1)
        self.layer_scale2 = nn.Parameter(torch.ones(1) * 0.1)
        self.layer_scale3 = nn.Parameter(torch.ones(1) * 0.1)
        
        # Output layers remain the same...
        self.output_configs = output_configs
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ff_dim * 3, ff_dim),
                nn.LayerNorm(ff_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(ff_dim, config['num_classes'])
            )
            for config in self.output_configs
        ])
        
        self.dropout_rate = dropout_rate
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch
        
        # Project input for first residual connection
        identity = self.proj1(x)
        
        # First attention block
        att_out = self.conv1(x, edge_index)  # Shape: [N, hidden_channels * heads]
        att_out = self.bn1(att_out)
        att_out = F.gelu(att_out)
        # Now both identity and att_out have shape [N, hidden_channels * heads]
        x = identity + self.layer_scale1 * F.dropout(att_out, p=self.dropout_rate, training=self.training)
        x = x + self.layer_scale1 * self.ff1(x)
        
        # Store for second residual
        identity = x
        
        # Second attention block
        att_out = self.conv2(x, edge_index)
        att_out = self.bn2(att_out)
        att_out = F.gelu(att_out)
        x = identity + self.layer_scale2 * F.dropout(att_out, p=self.dropout_rate, training=self.training)
        x = x + self.layer_scale2 * self.ff2(x)
        
        # Store for third residual
        identity = x
        
        # Third attention block
        att_out = self.conv3(x, edge_index)
        att_out = self.bn3(att_out)
        att_out = F.gelu(att_out)
        x = identity + self.layer_scale3 * F.dropout(att_out, p=self.dropout_rate, training=self.training)
        
        # Readout
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)
        x = torch.cat([mean_pool, max_pool, sum_pool], dim=1)
        
        # Output layers
        outputs = []
        for layer in self.output_layers:
            outputs.append(layer(x))
            
        return outputs

# Example dimensions:
"""
Assuming:
- in_channels = 64
- hidden_channels = 32
- heads = 8
- batch_size = 16

Shape transformations:
1. Input x: [N, 64]
2. After proj1: [N, 32 * 8] = [N, 256]
3. After conv1: [N, 32 * 8] = [N, 256]
4. After residual: [N, 256]
5. After ff1: [N, 256]
And so on...

where N is the number of nodes in the graph
"""