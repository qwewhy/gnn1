import torch
import torch.nn.functional as F
from pathlib import Path
import yaml
from typing import List, Tuple, Optional
import numpy as np

from src.models.gnn_encoder import DualInputGNNEncoder
from src.data_processing.pyg_dataset import PatchDataset


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
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else device)
        
        # 加载数据集 / Load dataset
        data_root = Path(self.config['data']['root'])
        self.dataset = PatchDataset(root=str(data_root))
        
        # 加载预计算的模式索引
        self.pattern_index = None
        index_path = data_root / 'processed' / 'pattern_index.pt'
        if index_path.exists():
            self.pattern_index = torch.load(index_path, map_location=self.device)
            print(f"Loaded pattern index with {self.pattern_index.shape[0]} embeddings.")
        else:
            print(f"Warning: Pattern index not found at {index_path}. Please run `build_index.py` first.")

        # 加载模型 / Load model
        self._load_model()
        
    def _load_model(self):
        """
        加载训练好的模型 / Load trained model.
        """
        model_config = self.config['model']
        self.model = DualInputGNNEncoder(
            anchor_in_channels=model_config['anchor_in_channels'],
            pattern_in_channels=model_config['pattern_in_channels'], 
            hidden_channels=model_config['hidden_channels'],
            out_channels=model_config['out_channels'],
            gnn_type=model_config['gnn_type']
        ).to(self.device)
        
        checkpoint_path = Path(self.config['training']['checkpoint_dir']) / 'best_model.pt'
        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded model from {checkpoint_path}")
        else:
            print(f"Warning: No model checkpoint found at {checkpoint_path}. The model is initialized with random weights.")

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
            return self.model(anchor_data.to(self.device), input_type='anchor')
    
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
            return self.model(pattern_data.to(self.device), input_type='pattern')
    
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
