# File: src/train/data_processing/triplet_generator.py
# 三元组生成模块 / Triplet generation module

import trimesh
import numpy as np
import networkx as nx
import torch
from typing import List, Optional, Dict, Union, Tuple
from pathlib import Path
import random

# 使用统一的路径管理器
from src.common.path_manager import path_manager
from src.train.data_processing.pyg_dataset import PatchDataset, Data


class TripletGenerator:
    """
    为度量学习生成三元组（锚点、正样本、负样本）
    """
    def __init__(self, mesh_paths: Union[str, List[str], Path, List[Path]],
                 patch_dataset: PatchDataset,
                 hard_mining_config: Optional[Dict] = None):

        if not isinstance(mesh_paths, list):
            mesh_paths = [mesh_paths]

        self.meshes = []
        for mesh_path in mesh_paths:
            # 统一处理路径，确保路径是绝对的
            mesh_path_obj = Path(mesh_path)
            if not mesh_path_obj.is_absolute():
                full_mesh_path = path_manager.get_model_path(str(mesh_path))
                if full_mesh_path.exists():
                    mesh_path = str(full_mesh_path)
            
            try:
                mesh = trimesh.load(mesh_path, process=True)
                self.meshes.append(mesh)
            except Exception as e:
                raise IOError(f"无法加载网格文件 at {mesh_path}: {e}")

        if not self.meshes:
            raise ValueError("未能加载任何有效的网格文件。")

        self.patch_dataset = patch_dataset
        self.hard_mining_config = hard_mining_config

        # 预计算面邻接图
        self.face_adjacency_graphs = [nx.from_edgelist(mesh.face_adjacency) for mesh in self.meshes]

    def extract_random_patch(self, mesh_index: int, min_faces: int = 10, max_faces: int = 20) -> Optional[List[int]]:
        """从指定网格中提取一个随机的、连通的面片"""
        mesh = self.meshes[mesh_index]
        face_adjacency_graph = self.face_adjacency_graphs[mesh_index]
        
        num_total_faces = len(mesh.faces)
        if num_total_faces < min_faces:
            return None

        for _ in range(10):  # 尝试10次
            start_face_idx = np.random.randint(0, num_total_faces)
            
            q = [start_face_idx]
            visited = {start_face_idx}
            patch_faces = [start_face_idx]

            while q and len(patch_faces) < max_faces:
                current_face = q.pop(0)
                for neighbor in face_adjacency_graph.neighbors(current_face):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        patch_faces.append(neighbor)
                        q.append(neighbor)
                        if len(patch_faces) >= max_faces:
                            break
            
            if min_faces <= len(patch_faces) <= max_faces:
                return patch_faces
        return None

    def _create_anchor_from_patch(self, patch_indices: List[int], mesh_index: int) -> Optional[Data]:
        """从面片索引创建一个PyG Data对象作为锚点"""
        mesh = self.meshes[mesh_index]
        try:
            # 提取面片的顶点和面
            patch_faces = mesh.faces[patch_indices]
            unique_vertices = np.unique(patch_faces)
            num_nodes = len(unique_vertices)
            
            # 创建顶点索引映射：原始索引 -> 局部索引
            vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
            
            # 构建面片内部的边
            edges = set()
            for face in patch_faces:
                # 为每个面添加边（假设是三角形面）
                for i in range(len(face)):
                    v1 = face[i]
                    v2 = face[(i + 1) % len(face)]
                    
                    # 只添加面片内部的边
                    if v1 in vertex_map and v2 in vertex_map:
                        # 重新映射到局部索引
                        local_v1 = vertex_map[v1]
                        local_v2 = vertex_map[v2]
                        edges.add(tuple(sorted((local_v1, local_v2))))
            
            # 创建双向边索引
            edge_list = []
            for v1, v2 in edges:
                edge_list.append([v1, v2])
                edge_list.append([v2, v1])
            
            if len(edge_list) == 0:
                # 如果没有边，创建一个简单的环
                edge_list = [[i, (i + 1) % num_nodes] for i in range(num_nodes)]
                edge_list.extend([[(i + 1) % num_nodes, i] for i in range(num_nodes)])
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            # 创建特征
            x = torch.randn(num_nodes, 8)  # 8维节点特征
            edge_attr = torch.randn(edge_index.shape[1], 3)  # 3维边特征

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        except Exception as e:
            print(f"创建锚点失败: {e}")
            return None

    def find_positive(self, anchor_canonical_form: str, anchor_id: int) -> Optional[Data]:
        """
        在数据集中找到一个与锚点具有相同规范形式但ID不同的正样本。
        """
        for i in range(len(self.patch_dataset)):
            sample = self.patch_dataset[i]
            if sample.canonical_form == anchor_canonical_form and sample.pattern_id != anchor_id:
                return sample
        return None

    def find_negative(self, anchor_canonical_form: str) -> Optional[Data]:
        """
        在数据集中找到一个与锚点具有不同规范形式的负样本。
        """
        # 为了高效，可以随机抽样
        indices = list(range(len(self.patch_dataset)))
        random.shuffle(indices)
        
        for i in indices:
            sample = self.patch_dataset[i]
            if sample.canonical_form != anchor_canonical_form:
                return sample
        return None

    def generate_triplets(self, num_triplets: int) -> List[Tuple[Data, Data, Data]]:
        """
        生成指定数量的三元组
        """
        triplets = []
        
        # 简化版：随机从数据集中抽取
        if len(self.patch_dataset) < 3:
            return []

        # 创建一个基于规范形式的字典
        canonical_map = {}
        for i in range(len(self.patch_dataset)):
            sample = self.patch_dataset[i]
            form = sample.canonical_form
            if form not in canonical_map:
                canonical_map[form] = []
            canonical_map[form].append(sample)
        
        canonical_forms = list(canonical_map.keys())

        while len(triplets) < num_triplets:
            # 1. 随机选择一个规范形式作为锚点和正样本的来源
            anchor_form = random.choice(canonical_forms)
            
            # 2. 如果该形式只有一个样本，无法构成正样本对，则跳过
            if len(canonical_map[anchor_form]) < 2:
                continue

            # 3. 从中随机选择一个锚点和一个正样本
            anchor, positive = random.sample(canonical_map[anchor_form], 2)

            # 4. 随机选择一个不同的规范形式作为负样本来源
            negative_form = random.choice(canonical_forms)
            while negative_form == anchor_form:
                 negative_form = random.choice(canonical_forms)

            # 5. 从中随机选择一个负样本
            negative = random.choice(canonical_map[negative_form])
            
            triplets.append((anchor, positive, negative))

        return triplets
