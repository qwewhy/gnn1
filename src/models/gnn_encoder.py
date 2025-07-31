import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data

class DualInputGNNEncoder(nn.Module):
    """
    双输入GNN编码器，处理几何查询和拓扑模式两种输入 / 
    Dual-input GNN encoder for geometric queries and topological patterns.
    
    使用孪生网络架构学习统一嵌入空间 / 
    Siamese network architecture for unified embedding space.
    """
    def __init__(self,
                 anchor_in_channels: int,
                 pattern_in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 gnn_type: str = 'gat'):
        """
        初始化双输入GNN编码器 / Initialize DualInputGNNEncoder.

        Args:
            anchor_in_channels (int): 锚点特征维度 / Anchor feature dimensions
            pattern_in_channels (int): 模式特征维度 / Pattern feature dimensions  
            hidden_channels (int): 隐藏层维度 / Hidden layer dimensions
            out_channels (int): 输出嵌入维度 / Output embedding dimensions
            gnn_type (str): GNN类型 / GNN type ('gcn', 'gat', 'graphsage')
        """
        super(DualInputGNNEncoder, self).__init__()

        # 输入投影层，映射到公共特征空间 / Input projection to common feature space
        self.anchor_proj = nn.Linear(anchor_in_channels, hidden_channels)
        self.pattern_proj = nn.Linear(pattern_in_channels, hidden_channels)

        self.gnn_type = gnn_type.lower()
        if self.gnn_type == 'gcn':
            self.conv1 = GCNConv(hidden_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif self.gnn_type == 'gat':
            self.conv1 = GATConv(hidden_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
        elif self.gnn_type == 'graphsage':
            self.conv1 = SAGEConv(hidden_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}. Choose from 'gcn', 'gat', 'graphsage'.")

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data: Data, input_type: str) -> torch.Tensor:
        """
        前向传播 / Forward pass through encoder.

        Args:
            data: 输入图数据 / Input graph data
            input_type (str): 输入类型 'anchor' 或 'pattern' / Input type 'anchor' or 'pattern'

        Returns:
            torch.Tensor: L2归一化的嵌入向量 / L2-normalized embedding vector
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. 根据输入类型选择投影层 / Apply projection based on input type
        if input_type == 'anchor':
            x = self.anchor_proj(x).relu()
        elif input_type == 'pattern':
            x = self.pattern_proj(x).relu()
        else:
            raise ValueError(f"Unknown input_type: {input_type}. Must be 'anchor' or 'pattern'.")

        # 2. 通过共享GNN层 / Pass through shared GNN layers
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        # 3. 全局池化 / Global pooling
        x = global_mean_pool(x, batch)

        # 4. 最终线性层和归一化 / Final linear layer and normalization
        x = self.lin(x)
        x = F.normalize(x, p=2, dim=1)

        return x
