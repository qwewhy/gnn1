# File: src/use/query_core.py
# 查询引擎模块 / Query engine module

import torch
from pathlib import Path
import yaml
from typing import List, Tuple

from src.train.models.improved_gat_encoder import MetricLearningGAT
from src.train.data_processing.pyg_dataset import PatchDataset


class QueryEngine:
    """
    查询引擎，用于几何模式相似性搜索 /
    Query engine for geometric pattern similarity search.
    """

    def __init__(self, config_path: str, device: str = 'auto'):
        """
        初始化查询引擎 / Initialize query engine.

        Args:
            config_path (str): YAML配置文件路径 / Path to YAML config file.
            device (str): 计算设备 / Computing device
        """
        self.project_root = Path(__file__).parent.parent.parent.absolute()

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device)

        # 加载数据集 / Load dataset
        data_root = self.project_root / self.config['data']['root']
        self.dataset = PatchDataset(root=str(data_root))

        # 加载预计算的模式索引
        self.pattern_index = None
        index_path = data_root / 'processed' / 'pattern_index.pt'
        if index_path.exists():
            self.pattern_index = torch.load(index_path, map_location=self.device, weights_only=False)
            print(f"Loaded pattern index with {self.pattern_index.shape[0]} embeddings.")
        else:
            print(f"Warning: Pattern index not found at {index_path}. Please run `build_index.py` first.")

        # 加载模型 / Load model - 修复模型配置传递问题
        self._load_model()

    def _load_model(self):
        """
        加载训练好的模型 / Load trained model.
        """
        # 修复：使用与训练脚本一致的配置结构
        encoder_config = {
            'anchor_in_channels': self.config['model']['anchor_in_channels'],
            'pattern_in_channels': self.config['model']['pattern_in_channels'],
            'hidden_channels': self.config['model']['hidden_channels'],
            'out_channels': self.config['model']['out_channels'],
            'num_heads': self.config['model'].get('num_heads', 4),
            'edge_dim': self.config['model'].get('edge_dim', 3),
            'dropout': self.config['model'].get('dropout', 0.2)
        }

        self.model = MetricLearningGAT(encoder_config).to(self.device)

        checkpoint_path = self.project_root / self.config['training']['checkpoint_dir'] / 'best_model.pt'
        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=False))
            self.model.eval()
            print(f"Loaded model from {checkpoint_path}")
        else:
            print(
                f"Warning: No model checkpoint found at {checkpoint_path}. The model is initialized with random weights.")

    def encode_anchor(self, anchor_data) -> torch.Tensor:
        """
        编码锚点几何查询 / Encode anchor geometric query.

        Args:
            anchor_data: 锚点图数据 / Anchor graph data

        Returns:
            torch.Tensor: 嵌入向量 / Embedding vector
        """
        self.model.eval()
        with torch.no_grad():
            return self.model.encoder(anchor_data.to(self.device), input_type='anchor')

    def encode_pattern(self, pattern_data) -> torch.Tensor:
        """
        编码拓扑模式 / Encode topological pattern.

        Args:
            pattern_data: 模式图数据 / Pattern graph data

        Returns:
            torch.Tensor: 嵌入向量 / Embedding vector
        """
        self.model.eval()
        with torch.no_grad():
            return self.model.encoder(pattern_data.to(self.device), input_type='pattern')

    def similarity_search(self, query_embedding: torch.Tensor,
                          k: int = 10) -> List[Tuple[int, float]]:
        """
        相似性搜索 / Similarity search.

        Args:
            query_embedding: 查询嵌入向量 / Query embedding
            k (int): 返回前k个结果 / Return top-k results

        Returns:
            List[Tuple[int, float]]: (索引, 相似度分数) / (index, similarity score)
        """
        if self.pattern_index is None:
            print("Error: Pattern index is not loaded. Cannot perform search.")
            return []

        # 计算余弦相似度
        # Cosine similarity is dot product of normalized embeddings
        similarities = torch.matmul(query_embedding, self.pattern_index.T).squeeze(0)

        # 获取top-k结果
        top_k_scores, top_k_indices = torch.topk(similarities, k=min(k, len(self.pattern_index)))

        return list(zip(top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()))

    def query(self, anchor_data, k: int = 10) -> List[Tuple[int, float]]:
        """
        执行查询 / Execute query.

        Args:
            anchor_data: 锚点查询数据 / Anchor query data
            k (int): 返回结果数量 / Number of results to return

        Returns:
            List[Tuple[int, float]]: 搜索结果 / Search results
        """
        # 编码查询 / Encode query
        query_emb = self.encode_anchor(anchor_data)

        # 相似性搜索 / Similarity search
        return self.similarity_search(query_emb, k)