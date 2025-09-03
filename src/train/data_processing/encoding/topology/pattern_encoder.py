# File: src/train/data_processing/encoding/topology/pattern_encoder.py
# 模式编码器 / Pattern encoder

import trimesh
from typing import List, Optional, Tuple
import logging
from .edgebreaker_encoder import EdgeBreakerEncoder


class ProperPatternEncoder:
    """模式编码器 - 直接使用EdgeBreaker"""
    def __init__(self):
        self.encoder = EdgeBreakerEncoder()
        
    def encode_patch_to_pattern(self, mesh: trimesh.Trimesh, 
                               patch_face_indices: List[int]) -> Optional[Tuple[str, int]]:
        """将几何面片编码为EdgeBreaker模式"""
        try:
            # 1. 直接使用EdgeBreaker编码
            encoding = self.encoder.encode_patch(mesh, patch_face_indices)
            
            if encoding is None:
                return None
                
            # 2. 计算边界边数
            num_sides = self._calculate_boundary_sides(mesh, patch_face_indices)
            
            if num_sides is None:
                return None
                
            return encoding, num_sides
            
        except Exception as e:
            logging.error(f"编码面片失败: {e}")
            return None
            
    def _calculate_boundary_sides(self, mesh: trimesh.Trimesh, 
                                 patch_face_indices: List[int]) -> Optional[int]:
        """计算面片的边界边数 - 改进版本，不依赖mesh.outline()"""
        try:
            # 使用和ImprovedPatchExtractor相同的方法
            edge_count = {}
            
            # 统计每条边被使用的次数
            for face_idx in patch_face_indices:
                face = mesh.faces[face_idx]
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                    edge_count[edge] = edge_count.get(edge, 0) + 1
            
            # 边界边只出现一次
            boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
            
            if not boundary_edges:
                return None
            
            # 构建边界图并追踪路径
            boundary_graph = {}
            for v1, v2 in boundary_edges:
                if v1 not in boundary_graph:
                    boundary_graph[v1] = []
                if v2 not in boundary_graph:
                    boundary_graph[v2] = []
                boundary_graph[v1].append(v2)
                boundary_graph[v2].append(v1)
            
            # 检查是否所有顶点都有度数2（形成简单环路）
            if not all(len(neighbors) == 2 for neighbors in boundary_graph.values()):
                return None
                
            # 边界顶点数等于边界边数
            return len(boundary_edges)
            
        except Exception as e:
            logging.error(f"计算边界边数失败: {e}")
            return None
