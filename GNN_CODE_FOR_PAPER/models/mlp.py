import torch.nn as nn
import torchmetrics
import torch
import torch.nn.functional as F

import lightning as L 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import torchmetrics
from torch.optim.lr_scheduler import MultiStepLR

class MLPDynamicMultiOutputModel(nn.Module):
    def __init__(self, input_shape, hidden_size,
                 intermediate_layers, output_shapes, 
                 dropout_rate, hidden_shape=1024, 
                 intermediate_shape=1024):
        super().__init__()
        self.norm = nn.BatchNorm1d(input_shape)
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(input_shape if i == 0 else hidden_shape, hidden_shape),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_shape),
                    nn.Dropout(dropout_rate)
                ) for i in range(hidden_size)
            ]
        )
        
        self.intermediate_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        
        for layers, output_shape in zip(intermediate_layers, output_shapes):
            intermediate = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(hidden_shape if i == 0 else intermediate_shape, intermediate_shape),
                        nn.ReLU(),
                        nn.BatchNorm1d(intermediate_shape),
                        nn.Dropout(dropout_rate)
                    ) for i in range(layers)
                ]
            )
            self.intermediate_layers.append(intermediate)
            self.output_layers.append(nn.Linear(intermediate_shape, output_shape))
        
        # self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        x = self.hidden_layers(x)
        
        outputs = []
        for intermediate, output in zip(self.intermediate_layers, self.output_layers):
            x_i = intermediate(x)
            # out_i = self.sigm(output(x_i))
            out_i = output(x_i) # logits
            outputs.append(out_i)
        
        return tuple(outputs)


class MLPDynamicMultiOutputModelV2(nn.Module):
    def __init__(self, input_shape, hidden_size,
                 intermediate_layers, output_shapes,
                 dropout_rate, hidden_shape=512, 
                 intermediate_shape=256):
        """
        Args:
            input_shape (int): Size of input features for one drug
            hidden_size (int): Number of hidden layers in shared network
            intermediate_layers (list): List of integers specifying number of layers for each output branch
            output_shapes (list): List of integers specifying output size for each branch
            dropout_rate (float): Dropout rate for all layers
            hidden_shape (int): Number of units in hidden layers
            intermediate_shape (int): Number of units in intermediate layers
        """
        super().__init__()
        self.norm = nn.BatchNorm1d(input_shape//2)
        
        # Shared hidden layers
        self.hidden_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(input_shape if i == 0 else hidden_shape, hidden_shape),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_shape),
                    nn.Dropout(dropout_rate)
                ) for i in range(hidden_size)
            ]
        )
        
        # Separate branches for each output
        self.intermediate_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        
        for layers, output_shape in zip(intermediate_layers, output_shapes):
            intermediate = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Linear(hidden_shape if i == 0 else intermediate_shape, intermediate_shape),
                        nn.ReLU(),
                        nn.BatchNorm1d(intermediate_shape),
                        nn.Dropout(dropout_rate)
                    ) for i in range(layers)
                ]
            )
            self.intermediate_layers.append(intermediate)
            self.output_layers.append(nn.Linear(intermediate_shape, output_shape))
        
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        drug_a_feat, drug_b_feat = torch.chunk(x, 2, dim=1)
        # Normalize inputs
        drug_a_feat = self.norm(drug_a_feat)
        drug_b_feat = self.norm(drug_b_feat)
        
        # Concatenate features for both directions
        a_to_b_combined = torch.cat([drug_a_feat, drug_b_feat], dim=1)
        b_to_a_combined = torch.cat([drug_b_feat, drug_a_feat], dim=1)
        
        # Process through shared hidden layers
        a_to_b_hidden = self.hidden_layers(a_to_b_combined)
        b_to_a_hidden = self.hidden_layers(b_to_a_combined)
        
        outputs = []
        # Process each output branch
        for intermediate, output in zip(self.intermediate_layers, self.output_layers):
            # Process both directions
            a_to_b_inter = intermediate(a_to_b_hidden)
            b_to_a_inter = intermediate(b_to_a_hidden)
            
            # Average predictions from both directions and apply softmax
            # mean_output = self.softmax(
            #     (output(a_to_b_inter) + output(b_to_a_inter)) / 2
            # )
            mean_output = (output(a_to_b_inter) + output(b_to_a_inter)) / 2
            outputs.append(mean_output)
        
        return tuple(outputs)





class MLPFlexibleDynamicLitMultiOutputModel(L.LightningModule):
    def __init__(self, model, 
                output_configs, 
                learning_rate, 
                weight_decay):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_configs = output_configs
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
            # Add metric tracking for logit norms
            self.train_logit_norms = torchmetrics.MeanMetric()
            self.val_logit_norms = torchmetrics.MeanMetric()

    def _calculate_logit_norms(self, logits):
            """
            Calculate L2 norms of logit vectors.
            
            Args:
                logits: List of logit tensors from different output heads
            Returns:
                mean_norm: Mean L2 norm across all outputs
            """
            norms = []
            for logit in logits:
                # Calculate L2 norm for each sample in the batch
                norm = torch.norm(logit, p=2, dim=1)  # L2 norm along class dimension
                norms.append(norm)
                
            # Combine norms from all outputs
            all_norms = torch.cat(norms)
            return all_norms.mean()

    def _shared_step(self, batch, batch_idx, metrics, stage):
        x_batch, *targets = batch
        predictions = self.model(x_batch)

        # # Calculate and log logit norms
        # mean_norm = self._calculate_logit_norms(predictions)
        # if stage == 'train':
        #     self.train_logit_norms(mean_norm)
        #     self.log('train_logit_norm', self.train_logit_norms,
        #             prog_bar=True, on_step=True, on_epoch=True)
        # else:  # validation
        #     self.val_logit_norms(mean_norm)
        #     self.log('val_logit_norm', self.val_logit_norms,
        #             prog_bar=True, on_step=False, on_epoch=True)
            
        # # Log per-head logit norms
        # for i, (pred, config) in enumerate(zip(predictions, self.output_configs)):
        #     head_norm = torch.norm(pred, p=2, dim=1).mean()
        #     self.log(f'{stage}_logit_norm_{config["name"]}', head_norm,
        #             prog_bar=False, on_step=False, on_epoch=True)

        total_loss = 0
        for i, (pred, target, config) in enumerate(zip(predictions, targets, self.output_configs)):
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
            
            self.log(f'{stage}_loss_{config["name"]}', loss, prog_bar=True)
            
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
        
        self.log(f'{stage}_total_loss', total_loss, prog_bar=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.train_metrics, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.val_metrics, 'val')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_batch = batch[0]
        predictions = self.model(x_batch)
        
        processed_predictions = []
        for pred, config in zip(predictions, self.output_configs):
            if config['classification_type'] == 'multi-class':
                # Apply softmax and get class probabilities
                pred_probs = torch.softmax(pred, dim=1)
                pred_indices = torch.argmax(pred_probs, dim=1)
            else:  # multi-label
                # Apply sigmoid and threshold
                pred_probs = torch.sigmoid(pred)
                pred_indices = (pred_probs > 0.5).float()
            
            processed_predictions.append({
                'probabilities': pred_probs,
                'predictions': pred_indices,
                'output_name': config['name']
            })
        
        return processed_predictions

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.learning_rate, 
                                      weight_decay=self.weight_decay)
        return optimizer
class LogitNormLoss(nn.Module):
    """
    Temperature: - Controls the peakiness of the softmax function
                 - Lower values make the distribution more peaked 
                 - Higher values make the distribution more uniform 
                 - Helps control overconfidence
                     .Small: More confident predictions
                     .Large: Less confident predictions
                - If you have overconfident predictions, 
                    you can increase the temperature(e.g., 0.01 - 5.0)
                    < 0.1: Make predictions more confident
                    0.1 - 1.0: More balanced predictions
                    1.0: Standard softmax
                    > 1.0: Less confident predictions(1.0 to 5.0)

    Eps: Small value to prevent division by zero

    Smoothing: Label smoothing factor
    Examples of smoothing label:
        - smoothing: 0.1
        - original label:[0, 0,1,  0] 
        - After smoothing:
            Correct class: 1 - 0.1(smoothing) = 0.9
            Other classes: 0.1(smoothing) / (num_classes - 1) = 0.1 / 3 = 0.0333
        - Final output: [0.0333, 0.0333, 0.9, 0.0333]
    """
    # def __init__(self, temperature=0.04, eps=1e-7, smoothing=0.2):
    def __init__(self, temperature, eps=0.0014, smoothing=0.2):
        super(LogitNormLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps
        self.smoothing = smoothing

    def forward(self, logits, labels):
        norm = torch.norm(logits, p=2, dim=1, keepdim=True)
        logits_normalized = logits / (norm + self.eps)
        logits_normalized = logits_normalized / self.temperature
        return F.cross_entropy(logits_normalized, labels, 
                               label_smoothing=self.smoothing)

class MLPFlexibleDynamicLitMultiOutputModelLogitNorm(L.LightningModule):
    def __init__(self, model, 
                output_configs, 
                learning_rate, 
                weight_decay,
                temperature=1000):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_configs = output_configs
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
                loss_fn = LogitNormLoss(temperature=temperature) 
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
            # Add metric tracking for logit norms
            self.train_logit_norms = torchmetrics.MeanMetric()
            self.val_logit_norms = torchmetrics.MeanMetric()

    def _calculate_logit_norms(self, logits):
            """
            Calculate L2 norms of logit vectors.
            
            Args:
                logits: List of logit tensors from different output heads
            Returns:
                mean_norm: Mean L2 norm across all outputs
            """
            norms = []
            for logit in logits:
                # Calculate L2 norm for each sample in the batch
                norm = torch.norm(logit, p=2, dim=1)  # L2 norm along class dimension
                norms.append(norm)
                
            # Combine norms from all outputs
            all_norms = torch.cat(norms)
            return all_norms.mean()

    def _shared_step(self, batch, batch_idx, metrics, stage):
        x_batch, *targets = batch
        predictions = self.model(x_batch)

        # Calculate and log logit norms
        mean_norm = self._calculate_logit_norms(predictions)
        if stage == 'train':
            self.train_logit_norms(mean_norm)
            self.log('train_logit_norm', self.train_logit_norms,
                    prog_bar=True, on_step=True, on_epoch=True)
        else:  # validation
            self.val_logit_norms(mean_norm)
            self.log('val_logit_norm', self.val_logit_norms,
                    prog_bar=True, on_step=False, on_epoch=True)
            
        # Log per-head logit norms
        for i, (pred, config) in enumerate(zip(predictions, self.output_configs)):
            head_norm = torch.norm(pred, p=2, dim=1).mean()
            self.log(f'{stage}_logit_norm_{config["name"]}', head_norm,
                    prog_bar=False, on_step=False, on_epoch=True)

        total_loss = 0
        for i, (pred, target, config) in enumerate(zip(predictions, targets, self.output_configs)):
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
            
            self.log(f'{stage}_loss_{config["name"]}', loss, prog_bar=True)
            
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
        
        self.log(f'{stage}_total_loss', total_loss, prog_bar=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.train_metrics, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, self.val_metrics, 'val')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x_batch = batch[0]
        predictions = self.model(x_batch)
        
        processed_predictions = []
        for pred, config in zip(predictions, self.output_configs):
            if config['classification_type'] == 'multi-class':
                # Apply softmax and get class probabilities
                pred_probs = torch.softmax(pred, dim=1)
                pred_indices = torch.argmax(pred_probs, dim=1)
            else:  # multi-label
                # Apply sigmoid and threshold
                pred_probs = torch.sigmoid(pred)
                pred_indices = (pred_probs > 0.5).float()
            
            processed_predictions.append({
                'probabilities': pred_probs,
                'predictions': pred_indices,
                'output_name': config['name']
            })
        
        return processed_predictions

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.learning_rate, 
                                      weight_decay=self.weight_decay)
        # Learning rate scheduler
        scheduler = MultiStepLR(optimizer, milestones=[80, 40], gamma=0.1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "interval": 'epoch',
                "frequency": 1
            } 
        }

def generate_mlp_output_config(*class_configs):
    total_classes = sum(config['num_classes'] for config in class_configs)
    
    output_configs = []
    for config in class_configs:
        name = config['name']
        num_classes = config['num_classes']
        classification_type = config['classification_type']
        intermediate_layers = config['intermediate_layers']
        loss_weight = num_classes / total_classes
        
        output_configs.append({
            'name': name,
            'num_classes': num_classes,
            'loss_weight': loss_weight,
            'classification_type': classification_type,
            'intermediate_layers': intermediate_layers,
            'target': f'{name}_y'
        })
    
    return output_configs

class MLPDynamicMultiOutputModelV3(nn.Module):
    def __init__(self, 
                 input_shape,
                 hidden_size,
                 intermediate_layers,
                 output_shapes,
                 dropout_rate=0.1,
                 hidden_shape=1024,
                 intermediate_shape=1024,
                 use_residual=True,
                 activation='gelu',
                 layer_norm=True,
                 bottleneck_factor=4):
        super().__init__()
        
        # Configuration
        self.use_residual = use_residual
        self.activation = self._get_activation(activation)
        
        # Input normalization and embedding
        self.input_norm = nn.LayerNorm(input_shape) if layer_norm else nn.BatchNorm1d(input_shape)
        self.input_projection = nn.Linear(input_shape, hidden_shape)
        
        # Main hidden layers with residual connections and layer normalization
        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_size):
            layer = EnhancedBlock(
                hidden_shape,
                dropout_rate,
                bottleneck_factor,
                use_residual,
                activation,
                layer_norm
            )
            self.hidden_layers.append(layer)
        
        # Task-specific intermediate and output layers
        self.task_networks = nn.ModuleList()
        for layers, output_shape in zip(intermediate_layers, output_shapes):
            task_net = TaskSpecificNetwork(
                input_dim=hidden_shape,
                output_dim=output_shape,
                intermediate_dim=intermediate_shape,
                num_layers=layers,
                dropout_rate=dropout_rate,
                bottleneck_factor=bottleneck_factor,
                use_residual=use_residual,
                activation=activation,
                layer_norm=layer_norm
            )
            self.task_networks.append(task_net)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _get_activation(self, activation):
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        return activations.get(activation.lower(), nn.GELU())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        # Input normalization and projection
        x = self.input_norm(x)
        x = self.input_projection(x)
        x = self.activation(x)
        
        # Apply hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Task-specific processing
        outputs = []
        for task_net in self.task_networks:
            out = task_net(x)
            outputs.append(out)
        
        return tuple(outputs)

class EnhancedBlock(nn.Module):
    """Enhanced block with bottleneck, residual connection, and normalization"""
    def __init__(self, 
                 dim, 
                 dropout_rate, 
                 bottleneck_factor=4, 
                 use_residual=True,
                 activation='gelu',
                 layer_norm=True):
        super().__init__()
        
        bottleneck_dim = dim // bottleneck_factor
        self.use_residual = use_residual
        self.activation = activation_fn = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'selu': nn.SELU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }[activation.lower()]
        
        # Two-layer bottleneck architecture
        self.net = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            activation_fn,
            nn.LayerNorm(bottleneck_dim) if layer_norm else nn.BatchNorm1d(bottleneck_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(bottleneck_dim, dim),
            activation_fn,
            nn.LayerNorm(dim) if layer_norm else nn.BatchNorm1d(dim),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        out = self.net(x)
        if self.use_residual:
            out = x + out
        return out

class TaskSpecificNetwork(nn.Module):
    """Task-specific network with enhanced architecture"""
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 intermediate_dim, 
                 num_layers,
                 dropout_rate=0.1,
                 bottleneck_factor=4,
                 use_residual=True,
                 activation='gelu',
                 layer_norm=True):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        # Progressive dimension reduction
        dims = self._get_progressive_dims(input_dim, intermediate_dim, num_layers)
        
        # Build layers with bottleneck architecture
        for i, dim in enumerate(dims):
            layers.append(EnhancedBlock(
                dim=current_dim,
                dropout_rate=dropout_rate,
                bottleneck_factor=bottleneck_factor,
                use_residual=use_residual,
                activation=activation,
                layer_norm=layer_norm
            ))
            if i < len(dims) - 1:  # Dimension reduction between blocks
                layers.append(nn.Linear(current_dim, dim))
                layers.append(nn.LayerNorm(dim) if layer_norm else nn.BatchNorm1d(dim))
                current_dim = dim
        
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, output_dim)
        
    def _get_progressive_dims(self, input_dim, target_dim, num_layers):
        """Calculate progressive dimension reduction"""
        if num_layers <= 1:
            return [target_dim]
        
        # Logarithmic spacing for smooth dimension reduction
        dims = np.logspace(np.log10(input_dim), np.log10(target_dim), num_layers)
        return [int(d) for d in dims[1:]]  # Skip first dim as it's handled by input
        
    def forward(self, x):
        x = self.network(x)
        return self.output_layer(x)