# File: src/train/models/improved_gat_encoder.py
# 改进的GAT编码器模型 / Improved GAT encoder model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
from typing import Optional, Tuple

class ImprovedGATEncoder(nn.Module):
    """
    改进的图注意力网络编码器，专门用于四边形化模式检索
    Enhanced GAT encoder specifically designed for quad pattern retrieval
    """
    
    def __init__(self,
                 anchor_in_channels: int = 8,  # 6维拓扑 + 2维几何
                 pattern_in_channels: int = 8,  # 6维拓扑 + 2维曲率
                 hidden_channels: int = 128,
                 out_channels: int = 128,
                 num_heads: int = 4,
                 edge_dim: int = 3,  # 边特征维度
                 dropout: float = 0.2):
        """
        初始化改进的GAT编码器
        
        Args:
            anchor_in_channels: 锚点（查询）特征维度
            pattern_in_channels: 模式特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出嵌入维度
            num_heads: 注意力头数
            edge_dim: 边特征维度
            dropout: Dropout比率
        """
        super(ImprovedGATEncoder, self).__init__()
        
        # 输入投影层 - 将不同输入映射到共同特征空间
        self.anchor_proj = nn.Sequential(
            nn.Linear(anchor_in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pattern_proj = nn.Sequential(
            nn.Linear(pattern_in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT层 - 使用多头注意力机制
        # 第一层：多头注意力，concat=True
        self.gat1 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            bias=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # 第二层：多头注意力
        self.gat2 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            bias=True
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # 第三层：单头注意力，用于最终特征
        self.gat3 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            heads=1,
            edge_dim=edge_dim,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
            bias=True
        )
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # 池化策略：结合多种池化方法
        self.pooling_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels * 2),  # 3种池化
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, out_channels)
        )
        
        # 可选：添加图级别特征的处理
        self.graph_feature_proj = nn.Linear(4, 32)  # 处理全局特征
        
        # 最终输出投影
        self.final_proj = nn.Sequential(
            nn.Linear(out_channels + 32, out_channels),
            nn.BatchNorm1d(out_channels)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data: Data, input_type: str) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: PyG数据对象
            input_type: 'anchor' 或 'pattern'
            
        Returns:
            L2归一化的嵌入向量
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # 1. 输入投影
        if input_type == 'anchor':
            x = self.anchor_proj(x)
        elif input_type == 'pattern':
            x = self.pattern_proj(x)
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
        
        # 2. GAT层传播
        # 第一层GAT
        x1 = self.gat1(x, edge_index, edge_attr=edge_attr)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)
        
        # 残差连接
        x = x + x1 if x.shape == x1.shape else x1
        
        # 第二层GAT
        x2 = self.gat2(x, edge_index, edge_attr=edge_attr)
        x2 = self.bn2(x2)
        x2 = F.elu(x2)
        x2 = self.dropout(x2)
        
        # 残差连接
        x = x + x2 if x.shape == x2.shape else x2
        
        # 第三层GAT
        x3 = self.gat3(x, edge_index, edge_attr=edge_attr)
        x3 = self.bn3(x3)
        x3 = F.elu(x3)
        
        # 3. 多策略池化
        # 平均池化 - 捕获整体特征
        x_mean = global_mean_pool(x3, batch)
        # 最大池化 - 捕获显著特征（如奇异点）
        x_max = global_max_pool(x3, batch)
        # 求和池化 - 保留总体信息
        x_sum = global_add_pool(x3, batch)
        
        # 连接不同池化结果
        x_pooled = torch.cat([x_mean, x_max, x_sum], dim=1)
        
        # 4. 通过MLP处理池化特征
        x_embed = self.pooling_mlp(x_pooled)
        
        # 5. 可选：整合图级别特征
        if hasattr(data, 'u') and data.u is not None:
            # 处理图级别特征（如复杂度分数、平均曲率等）
            graph_features = self.graph_feature_proj(data.u)
            x_embed = torch.cat([x_embed, graph_features], dim=1)
            x_embed = self.final_proj(x_embed)
        
        # 6. L2归一化
        x_embed = F.normalize(x_embed, p=2, dim=1)
        
        return x_embed
    
    def get_attention_weights(self, data: Data, input_type: str, layer: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取注意力权重用于可视化
        
        Args:
            data: 输入数据
            input_type: 输入类型
            layer: 要获取注意力权重的层 (1, 2, 或 3)
            
        Returns:
            (edge_index, attention_weights)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # 输入投影
        if input_type == 'anchor':
            x = self.anchor_proj(x)
        else:
            x = self.pattern_proj(x)
        
        # 根据请求的层获取注意力权重
        if layer == 1:
            _, (edge_index_out, alpha) = self.gat1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        elif layer == 2:
            x = F.elu(self.bn1(self.gat1(x, edge_index, edge_attr=edge_attr)))
            _, (edge_index_out, alpha) = self.gat2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        else:  # layer == 3
            x = F.elu(self.bn1(self.gat1(x, edge_index, edge_attr=edge_attr)))
            x = F.elu(self.bn2(self.gat2(x, edge_index, edge_attr=edge_attr)))
            _, (edge_index_out, alpha) = self.gat3(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        
        return edge_index_out, alpha


class MetricLearningGAT(nn.Module):
    """
    完整的度量学习GAT模型，包含损失计算
    Complete metric learning GAT model with loss computation
    """
    
    def __init__(self, encoder_config: dict):
        super(MetricLearningGAT, self).__init__()
        self.encoder = ImprovedGATEncoder(**encoder_config)
        
    def forward_triplet(self, anchor: Data, positive: Data, negative: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理三元组输入
        
        Returns:
            (anchor_emb, positive_emb, negative_emb)
        """
        anchor_emb = self.encoder(anchor, input_type='anchor')
        positive_emb = self.encoder(positive, input_type='pattern')
        negative_emb = self.encoder(negative, input_type='pattern')
        
        return anchor_emb, positive_emb, negative_emb
    
    def compute_triplet_loss(self, anchor_emb: torch.Tensor, 
                           positive_emb: torch.Tensor, 
                           negative_emb: torch.Tensor,
                           margin: float = 0.5) -> torch.Tensor:
        """
        计算三元组损失，包含难负例挖掘
        """
        # 计算距离
        pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
        neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)
        
        # 三元组损失
        loss = F.relu(pos_dist - neg_dist + margin)
        
        # 只对有效的三元组计算损失（难负例）
        hard_triplets = loss > 0
        if hard_triplets.sum() > 0:
            loss = loss[hard_triplets].mean()
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        计算对比损失（可选的辅助损失）
        """
        # 归一化嵌入
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # 创建标签矩阵
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 计算对比损失
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 只对正样本对计算损失
        loss = -(mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return loss.mean()
