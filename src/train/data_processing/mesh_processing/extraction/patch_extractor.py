# File: src/train/data_processing/mesh_processing/extraction/patch_extractor.py
# 面片提取器 / Patch extractor

import trimesh
import numpy as np
import networkx as nx
from typing import List, Optional
from collections import deque, defaultdict
import logging


class ImprovedPatchExtractor:
    """改进的面片提取器，增加了多项验证"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_valid_patch(self, mesh: trimesh.Trimesh, 
                           min_faces: int = 10, max_faces: int = 20,
                           max_attempts: int = 50) -> Optional[List[int]]:
        """提取几何和拓扑上都有效的面片"""
        
        # 预处理网格
        if not self._is_mesh_valid(mesh):
            self.logger.warning("网格无效，跳过处理")
            return None
            
        face_adjacency_graph = self._build_robust_face_adjacency(mesh)
        
        for attempt in range(max_attempts):
            patch_faces = self._extract_single_patch(
                mesh, face_adjacency_graph, min_faces, max_faces
            )
            
            if patch_faces is None:
                continue
                
            # 多重验证
            if self._validate_patch_basic(mesh, patch_faces):
                self.logger.info(f"成功提取有效patch，尝试次数: {attempt + 1}")
                return patch_faces
                
        self.logger.warning(f"经过{max_attempts}次尝试，未能提取到有效patch")
        return None
    
    def _is_mesh_valid(self, mesh: trimesh.Trimesh) -> bool:
        """验证网格基本有效性"""
        try:
            return (
                len(mesh.faces) >= 20 and  # 足够的面
                len(mesh.vertices) >= 10 and  # 足够的顶点
                not mesh.is_empty  # 非空
            )
        except:
            # 如果验证失败，进行基本检查
            return len(mesh.faces) >= 20 and len(mesh.vertices) >= 10
    
    def _build_robust_face_adjacency(self, mesh: trimesh.Trimesh) -> nx.Graph:
        """构建鲁棒的面邻接图"""
        try:
            # 使用trimesh内建的面邻接关系
            face_adjacency = mesh.face_adjacency
            graph = nx.Graph()
            graph.add_edges_from(face_adjacency)
            return graph
        except Exception as e:
            self.logger.error(f"构建面邻接图失败: {e}")
            # 备用方法：手动构建
            return self._manual_face_adjacency(mesh)
    
    def _manual_face_adjacency(self, mesh: trimesh.Trimesh) -> nx.Graph:
        """手动构建面邻接图（备用方法）"""
        graph = nx.Graph()
        faces = mesh.faces
        
        # 构建边到面的映射
        edge_to_faces = {}
        for face_idx, face in enumerate(faces):
            for i in range(len(face)):
                edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
        
        # 添加邻接关系
        for edge, face_list in edge_to_faces.items():
            if len(face_list) == 2:  # 共享边的两个面
                graph.add_edge(face_list[0], face_list[1])
        
        return graph
    
    def _extract_single_patch(self, mesh: trimesh.Trimesh, 
                             face_adjacency_graph: nx.Graph,
                             min_faces: int, max_faces: int) -> Optional[List[int]]:
        """提取单个面片（BFS）"""
        num_total_faces = len(mesh.faces)
        if num_total_faces < min_faces:
            return None
        
        # 选择连通性好的起始面
        start_face_idx = self._select_good_start_face(
            mesh, face_adjacency_graph, num_total_faces
        )
        
        # BFS扩展
        queue = deque([start_face_idx])
        visited = {start_face_idx}
        patch_faces = [start_face_idx]
        
        while queue and len(patch_faces) < max_faces:
            current_face = queue.popleft()
            
            neighbors = list(face_adjacency_graph.neighbors(current_face))
            # 按某种策略排序邻居（比如面积、法向量相似度）
            neighbors = self._sort_neighbors_by_quality(mesh, current_face, neighbors)
            
            for neighbor in neighbors:
                if neighbor not in visited and len(patch_faces) < max_faces:
                    visited.add(neighbor)
                    patch_faces.append(neighbor)
                    queue.append(neighbor)
        
        return patch_faces if len(patch_faces) >= min_faces else None
    
    def _select_good_start_face(self, mesh: trimesh.Trimesh, 
                               graph: nx.Graph, num_faces: int) -> int:
        """选择连通性好的起始面"""
        # 倾向选择度数适中的面作为起始点
        degrees = dict(graph.degree())
        
        # 过滤掉度数过低或过高的面
        good_faces = [
            face_idx for face_idx, degree in degrees.items() 
            if 2 <= degree <= 6
        ]
        
        if good_faces:
            return np.random.choice(good_faces)
        else:
            return np.random.randint(0, num_faces)
    
    def _sort_neighbors_by_quality(self, mesh: trimesh.Trimesh, 
                                  current_face: int, neighbors: List[int]) -> List[int]:
        """按质量排序邻居面"""
        if not neighbors:
            return neighbors
            
        try:
            current_normal = mesh.face_normals[current_face]
            
            # 计算法向量相似度
            scores = []
            for neighbor in neighbors:
                neighbor_normal = mesh.face_normals[neighbor]
                similarity = np.dot(current_normal, neighbor_normal)
                scores.append((neighbor, similarity))
            
            # 按相似度降序排序
            scores.sort(key=lambda x: x[1], reverse=True)
            return [face_idx for face_idx, _ in scores]
            
        except:
            return neighbors  # 如果计算失败，返回原顺序
    
    def _validate_patch_basic(self, mesh: trimesh.Trimesh, 
                             patch_faces: List[int]) -> bool:
        """基本验证patch的有效性"""
        
        # 1. 基本检查
        if not patch_faces or len(patch_faces) < 3:
            return False
        
        # 2. 检查面索引有效性
        max_face_idx = len(mesh.faces) - 1
        if any(face_idx < 0 or face_idx > max_face_idx for face_idx in patch_faces):
            return False
        
        return True
    
    def _extract_single_patch_with_graph(self, mesh: trimesh.Trimesh, 
                                        face_adjacency_graph: nx.Graph,
                                        min_faces: int, max_faces: int) -> Optional[List[int]]:
        """使用外部提供的面邻接图提取单个面片（与原始版本兼容）"""
        num_total_faces = len(mesh.faces)
        if num_total_faces < min_faces:
            return None
        
        # 选择连通性好的起始面
        start_face_idx = self._select_good_start_face(
            mesh, face_adjacency_graph, num_total_faces
        )
        
        # BFS扩展
        queue = deque([start_face_idx])
        visited = {start_face_idx}
        patch_faces = [start_face_idx]
        
        while queue and len(patch_faces) < max_faces:
            current_face = queue.popleft()
            
            neighbors = list(face_adjacency_graph.neighbors(current_face))
            # 按某种策略排序邻居（比如面积、法向量相似度）
            neighbors = self._sort_neighbors_by_quality(mesh, current_face, neighbors)
            
            for neighbor in neighbors:
                if neighbor not in visited and len(patch_faces) < max_faces:
                    visited.add(neighbor)
                    patch_faces.append(neighbor)
                    queue.append(neighbor)
        
        return patch_faces if len(patch_faces) >= min_faces else None


def extract_random_patch(mesh: trimesh.Trimesh, face_adjacency_graph: nx.Graph,
                         min_faces: int = 10, max_faces: int = 20) -> Optional[List[int]]:
    """使用改进的面片提取器 - 与原始版本兼容"""
    extractor = ImprovedPatchExtractor()
    # 直接使用传入的face_adjacency_graph，而不是重新构建
    return extractor._extract_single_patch_with_graph(mesh, face_adjacency_graph, min_faces, max_faces)
