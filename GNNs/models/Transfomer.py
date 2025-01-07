import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import LayerNorm, Linear, NNConv, GCNConv
import torch.nn.functional as F

class Convert2Dto3DWithPadding(nn.Module):
    def __init__(self):
        super(Convert2Dto3DWithPadding, self).__init__()

    def forward(self, batch: Batch):
        """
        Convert a PyTorch Geometric Batch of 2D chemical graph data 
        to a 3D tensor with padding.
        
        Args:
        batch (Batch): A PyTorch Geometric Batch object containing chemical graph data.
        
        Returns:
        torch.Tensor: A 3D tensor [num_graphs, max_num_atoms, num_features]
        torch.Tensor: A mask tensor indicating valid (1) and padded (0) atoms
        """
        # Extract necessary information
        num_graphs = batch.num_graphs
        num_features = batch.num_features
        
        # Get the number of atoms for each graph in the batch
        num_atoms_per_graph = torch.bincount(batch.batch)
        max_num_atoms = num_atoms_per_graph.max().item()
        
        # Create empty tensors for the result and mask
        result = torch.zeros(num_graphs, max_num_atoms, num_features, device=batch.x.device)
        mask = torch.zeros(num_graphs, max_num_atoms, dtype=torch.bool, device=batch.x.device)
        
        # Fill the result tensor and create the mask
        start = 0
        for i, num_atoms in enumerate(num_atoms_per_graph):
            end = start + num_atoms
            result[i, :num_atoms] = batch.x[start:end]
            mask[i, :num_atoms] = 1
            start = end
        
        return result, mask

class ChemicalGraphTransformer(nn.Module):
    def __init__(self, num_features, nhead, num_encoder_layers, 
                 dim_feedforward, dropout=0.1):
        super(ChemicalGraphTransformer, self).__init__()
        
        self.positional_encoding = PositionalEncoding(num_features, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_features, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        self.output_layer = nn.Linear(num_features, 1)  # For regression task
        
    def forward(self, x, mask):
        # x shape: [batch_size, max_num_atoms, num_features]
        # mask shape: [batch_size, max_num_atoms]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Create attention mask from padding mask
        attn_mask = ~mask
        
        # Apply Transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=attn_mask)
        
        # Global average pooling over atoms
        output = torch.sum(output * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1, keepdim=True)
        
        # Final output layer
        return output

# class ChemicalGraphTransformer(nn.Module):
#     def __init__(self, num_features, nhead, num_encoder_layers, 
#                  dim_feedforward, dropout=0.1):
#         super(ChemicalGraphTransformer, self).__init__()
        
#         self.positional_encoding = PositionalEncoding(num_features, dropout)
        
#         # Modified to use a more stable implementation
#         self.layer_norm = nn.LayerNorm(num_features)
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=num_features, 
#             nhead=nhead, 
#             dim_feedforward=dim_feedforward, 
#             dropout=dropout,
#             batch_first=True,
#             norm_first=True  # Apply normalization first for better stability
#         )
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer, 
#             num_layers=num_encoder_layers,
#             norm=self.layer_norm
#         )
        
#         self.output_layer = nn.Linear(num_features, 1)
        
#     def forward(self, x, mask):
#         # x shape: [batch_size, max_num_atoms, num_features]
#         # mask shape: [batch_size, max_num_atoms]
        
#         # Add positional encoding
#         x = self.positional_encoding(x)
        
#         # Create attention mask for valid positions
#         # Convert boolean mask to float attention mask
#         attn_mask = mask.float().masked_fill(
#             mask == 0,
#             float('-inf')
#         ).masked_fill(
#             mask == 1,
#             float(0.0)
#         )
        
#         # Apply Transformer encoder with attention mask
#         # Note: we use attention mask instead of key_padding_mask
#         output = self.transformer_encoder(x, mask=attn_mask.unsqueeze(1).unsqueeze(2))
        
#         # Apply mask for proper averaging
#         mask_expanded = mask.unsqueeze(-1).float()
#         output = output * mask_expanded
        
#         # Global average pooling over atoms (masked)
#         # Add small epsilon to avoid division by zero
#         output = torch.sum(output, dim=1) / (mask.sum(dim=1, keepdim=True).float() + 1e-9)
        
#         return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class NNConvLayerTransformer(nn.Module):
    def __init__(self, 
                 num_node_features, 
                 num_edge_features, 
                 dropout_rate,
                 expand_factor,
                 output_configs,
                 nhead,
                 num_encoder_layers,):

        super().__init__()
        self.output_configs = output_configs
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        conv1_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_node_features*num_node_features*expand_factor))
        conv2_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_node_features*expand_factor *num_node_features))
        self.conv1 = NNConv(num_node_features, num_node_features*expand_factor, 
                            conv1_net)
        self.conv2 = NNConv(num_node_features*expand_factor, num_node_features, 
                            conv2_net)
        self.transformer_encoder = ChemicalGraphTransformer(
                                num_features=num_node_features, 
                                nhead=self.nhead, 
                                num_encoder_layers=self.num_encoder_layers, 
                                dim_feedforward=num_node_features*expand_factor, 
                                dropout=dropout_rate)
        self.fc_1 = nn.Linear(num_node_features*2, num_node_features*expand_factor)
        # self.out = nn.Linear(num_node_features*expand_factor, NUM_DESC_CLASSES)
        self.output_layers = nn.ModuleList([
            torch.nn.Linear(num_node_features*expand_factor, 
                            config['num_classes'])
            for config in self.output_configs
        ])
        self.convert_2d_to_3d = Convert2Dto3DWithPadding()
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index,\
              data.edge_attr, data.batch
        x = F.relu(self.conv1(x.float(), edge_index, edge_attr.float()))
        x = F.relu(self.conv2(x, edge_index, edge_attr.float()))
        x_1 = global_add_pool(x, batch)

        # Transformer
        data.x = x
        x_3d, mask = self.convert_2d_to_3d(data)
        x_t = self.transformer_encoder(x_3d, mask)
        x_c = torch.cat((x_1, x_t), dim=1)
        x_c = F.relu(self.fc_1(x_c))
        # output = self.out(output)
        outputs = []
        for layer, config in zip(self.output_layers, self.output_configs):
            out = layer(x_c) # logits
            outputs.append(out)
        return outputs
class ECNTransformer(nn.Module):
    def __init__(self, 
                 num_node_features, 
                 num_edge_features, 
                 dropout_rate,
                 expand_factor,
                 output_configs,
                 nhead,
                 num_encoder_layers,):
        super().__init__()
        self.output_configs = output_configs
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        
        # First conv layer: num_node_features -> num_node_features*expand_factor
        conv1_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_node_features * (num_node_features*expand_factor))
        )
        
        # Second conv layer: num_node_features*expand_factor -> num_node_features*expand_factor
        conv2_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, (num_node_features*expand_factor) * (num_node_features*expand_factor))
        )
        
        # Third conv layer: num_node_features*expand_factor -> num_node_features
        conv3_net = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, (num_node_features*expand_factor) * num_node_features)
        )
        
        # NNConv layers with correct input/output dimensions
        self.conv1 = NNConv(num_node_features, num_node_features*expand_factor, conv1_net)
        self.conv2 = NNConv(num_node_features*expand_factor, num_node_features*expand_factor, conv2_net)
        self.conv3 = NNConv(num_node_features*expand_factor, num_node_features, conv3_net)
        
        self.transformer_encoder = ChemicalGraphTransformer(
                                num_features=num_node_features,  # Now correct as conv3 outputs num_node_features
                                nhead=self.nhead, 
                                num_encoder_layers=self.num_encoder_layers, 
                                dim_feedforward=num_node_features*expand_factor, 
                                dropout=dropout_rate)
        
        self.fc_1 = nn.Linear(num_node_features*2, num_node_features*expand_factor)
        self.output_layers = nn.ModuleList([
            torch.nn.Linear(num_node_features*expand_factor, config['num_classes'])
            for config in self.output_configs
        ])
        self.convert_2d_to_3d = Convert2Dto3DWithPadding()
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Forward passes with shape comments
        x = F.relu(self.conv1(x.float(), edge_index, edge_attr.float()))  # [num_nodes, num_node_features*expand_factor]
        x = F.relu(self.conv2(x, edge_index, edge_attr.float()))         # [num_nodes, num_node_features*expand_factor]
        x = F.relu(self.conv3(x, edge_index, edge_attr.float()))         # [num_nodes, num_node_features]
        
        x_1 = global_add_pool(x, batch)  # [batch_size, num_node_features]
        data.x = x
        x_3d, mask = self.convert_2d_to_3d(data)
        x_t = self.transformer_encoder(x_3d, mask)  # [batch_size, num_node_features]
        x_c = torch.cat((x_1, x_t), dim=1)  # [batch_size, num_node_features*2]
        x_c = F.relu(self.fc_1(x_c))  # [batch_size, num_node_features*expand_factor]
        
        outputs = []
        for layer, config in zip(self.output_layers, self.output_configs):
            out = layer(x_c)  # [batch_size, num_classes]
            outputs.append(out)
        return outputs

class GCNTransformer(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 output_configs, 
                 dropout_rate,
                 nhead=8,
                 num_encoder_layers=3,
                 expand_factor=2):
        super(GCNTransformer, self).__init__()
        self.output_configs = output_configs
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        
        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        
        # Transformer encoder
        self.transformer_encoder = ChemicalGraphTransformer(
            num_features=hidden_channels,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=hidden_channels*expand_factor,
            dropout=dropout_rate
        )
        
        # 2D to 3D conversion layer
        self.convert_2d_to_3d = Convert2Dto3DWithPadding()
        
        # Fusion layer
        self.fusion_layer = nn.Linear(hidden_channels * 4, hidden_channels * expand_factor)
        
        # Output layers
        self.output_layers = nn.ModuleList([
            torch.nn.Linear(hidden_channels * expand_factor, config['num_classes'])
            for config in self.output_configs
        ])
        
        self.dropout_rate = dropout_rate
    
    def forward(self, data):
        # 1. Obtain node embeddings through GCN
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        prev_x = x
        
        # Second GCN layer with residual connection
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x + prev_x
        x = F.relu(x)
        prev_x = x
        
        # Third GCN layer with residual connection
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x + prev_x
        x = F.relu(x)
        
        # 2. Global pooling (stack different aggregations)
        x_pooled = torch.cat([
            global_mean_pool(x, batch),
            global_add_pool(x, batch),
            global_max_pool(x, batch)
        ], dim=1)
        
        # 3. Transform through Transformer
        data.x = x  # Update the node features
        x_3d, mask = self.convert_2d_to_3d(data)
        x_transformer = self.transformer_encoder(x_3d, mask)
        
        # 4. Combine GCN and Transformer features
        x_combined = torch.cat([x_pooled, x_transformer], dim=1)
        x_combined = self.fusion_layer(x_combined)
        x_combined = F.relu(x_combined)
        
        # 5. Apply dropout
        x_combined = F.dropout(x_combined, p=self.dropout_rate, training=self.training)
        
        # 6. Apply multiple output layers
        outputs = []
        for layer, config in zip(self.output_layers, self.output_configs):
            out = layer(x_combined)
            outputs.append(out)
            
        return outputs