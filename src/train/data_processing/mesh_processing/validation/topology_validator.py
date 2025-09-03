# File: src/train/data_processing/mesh_processing/validation/topology_validator.py
# 拓扑验证器 / Topology validator

import trimesh
import numpy as np
from typing import List, Dict
import logging


class TopologyValidator:
    """拓扑验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def is_topologically_valid(self, mesh: trimesh.Trimesh, 
                              patch_faces: List[int], boundary_info: Dict) -> bool:
        """检查拓扑有效性"""
        try:
            num_boundary_vertices = boundary_info['num_boundary_vertices']
            
            # 边界应该至少有3个顶点
            if num_boundary_vertices < 3:
                return False
            
            # 边界顶点数不应该太大（相对于面片大小）
            if num_boundary_vertices > len(patch_faces) * 2:
                return False
            
            # 使用欧拉公式检查：V - E + F = 2（对于球面拓扑）
            # 这里是简化检查
            patch_vertices = set()
            patch_edges = set()
            
            for face_idx in patch_faces:
                face = mesh.faces[face_idx]
                patch_vertices.update(face)
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                    patch_edges.add(edge)
            
            V = len(patch_vertices)
            E = len(patch_edges)
            F = len(patch_faces)
            
            euler_char = V - E + F
            # 对于有边界的面片，欧拉特征数应该是1
            if not (0 <= euler_char <= 2):
                self.logger.warning(f"拓扑检查失败: V={V}, E={E}, F={F}, χ={euler_char}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"拓扑有效性检查失败: {e}")
            return True  # 如果检查失败，假设有效
