# File: src/train/data_processing/mesh_processing/validation/geometry_validator.py
# 几何验证器 / Geometry validator

import trimesh
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple, Set
import logging


class GeometryValidator:
    """几何验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def is_geometrically_connected(self, mesh: trimesh.Trimesh,
                                  patch_faces: List[int]) -> bool:
        """检查面片是否几何连通"""
        try:
            # 提取patch的顶点
            patch_vertices = set()
            for face_idx in patch_faces:
                face = mesh.faces[face_idx]
                patch_vertices.update(face)
            
            # 构建patch内部的顶点连通图
            vertex_graph = nx.Graph()
            for face_idx in patch_faces:
                face = mesh.faces[face_idx]
                for i in range(len(face)):
                    v1, v2 = face[i], face[(i + 1) % len(face)]
                    vertex_graph.add_edge(v1, v2)
            
            # 检查连通性
            return nx.is_connected(vertex_graph)
            
        except Exception as e:
            self.logger.warning(f"几何连通性检查失败: {e}")
            return True  # 如果检查失败，假设连通
    
    def compute_patch_boundary(self, mesh: trimesh.Trimesh, 
                              patch_faces: List[int]) -> Optional[Dict]:
        """计算面片边界信息"""
        try:
            patch_face_set = set(patch_faces)
            boundary_edges = []
            
            # 找出边界边（只属于一个面的边）
            edge_count = {}
            for face_idx in patch_faces:
                face = mesh.faces[face_idx]
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                    edge_count[edge] = edge_count.get(edge, 0) + 1
            
            # 边界边只出现一次
            boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
            
            if not boundary_edges:
                return None
            
            # 构建边界路径
            boundary_vertices = self._trace_boundary_path(boundary_edges)
            
            return {
                'edges': boundary_edges,
                'vertices': boundary_vertices,
                'num_boundary_vertices': len(boundary_vertices) if boundary_vertices else 0
            }
            
        except Exception as e:
            self.logger.error(f"边界计算失败: {e}")
            return None
    
    def _trace_boundary_path(self, boundary_edges: List[Tuple[int, int]]) -> Optional[List[int]]:
        """追踪边界路径，形成有序的边界顶点序列"""
        if not boundary_edges:
            return None
            
        # 构建边界图
        boundary_graph = nx.Graph()
        boundary_graph.add_edges_from(boundary_edges)
        
        # 检查是否形成简单环路
        if not all(degree == 2 for _, degree in boundary_graph.degree()):
            self.logger.warning("边界不形成简单环路")
            return None
        
        # 追踪路径
        try:
            start_vertex = boundary_edges[0][0]
            path = [start_vertex]
            current = start_vertex
            prev = None
            
            while True:
                neighbors = [n for n in boundary_graph.neighbors(current) if n != prev]
                if not neighbors:
                    break
                    
                next_vertex = neighbors[0]
                if next_vertex == start_vertex:  # 回到起点
                    break
                    
                path.append(next_vertex)
                prev = current
                current = next_vertex
                
                # 防止无限循环
                if len(path) > len(boundary_edges) + 1:
                    break
            
            return path if len(path) >= 3 else None
            
        except Exception as e:
            self.logger.error(f"边界路径追踪失败: {e}")
            return None
