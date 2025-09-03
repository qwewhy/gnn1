# File: src/train/data_processing/encoding/topology/pattern_decoder.py
# 模式解码器 / Pattern decoder

import torch
from typing import Dict
from .edgebreaker_decoder import EdgebreakerDecoder


class ProperPatternParser:
    """模式解析器"""
    def __init__(self, pattern_string: str, sides: int):
        self.pattern_string = pattern_string
        self.sides = sides
        self.decoder = EdgebreakerDecoder()
        
    def parse(self) -> Dict:
        """解析模式字符串生成图拓扑"""
        graph_data = self.decoder.decode_pattern_string(self.pattern_string, self.sides)
        
        if graph_data is None:
            graph_data = self.decoder._create_fallback_graph(self.sides)
        
        node_features = graph_data["node_features"]
        
        return {
            "edge_index": graph_data["edge_index"],
            "num_nodes": graph_data["num_nodes"],
            "node_valence": node_features["node_valence"],
            "is_boundary_node": node_features["is_boundary_node"],
            "is_corner_node": node_features["is_corner_node"],
            "distance_to_singular": node_features["distance_to_singular"],
            "local_topology_config": node_features["local_topology_config"],
            "boundary_position_encoding": node_features["boundary_position_encoding"],
            "is_boundary_edge": graph_data["edge_features"],
            "boundary_edges": graph_data["boundary_edges"] # 传递边界边信息
        }
    
    def _create_boundary_fallback(self) -> Dict:
        """创建基本边界环作为后备方案"""
        num_nodes = self.sides
        edges = []
        for i in range(num_nodes):
            edges.append([i, (i + 1) % num_nodes])
            edges.append([(i + 1) % num_nodes, i])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return {
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "node_valence": torch.full((num_nodes,), 2, dtype=torch.long),
            "is_boundary_node": torch.ones(num_nodes, dtype=torch.bool),
            "is_corner_node": torch.zeros(num_nodes, dtype=torch.bool),
            "distance_to_singular": torch.zeros(num_nodes, dtype=torch.float),
            "local_topology_config": torch.zeros(num_nodes, dtype=torch.float),
            "boundary_position_encoding": torch.arange(num_nodes, dtype=torch.float) / num_nodes,
            "is_boundary_edge": torch.ones(edge_index.shape[1], dtype=torch.bool)
        }
