# File: src/train/data_processing/pyg_dataset.py
# PyG数据集模块 / PyG dataset module

import sqlite3
import torch
from torch_geometric.data import InMemoryDataset, Data
from pathlib import Path
import tqdm
import json
import numpy as np

# 使用集中的路径管理器
from src.common.path_manager import path_manager
from src.train.data_processing.encoding.topology.pattern_decoder import ProperPatternParser


class PatchDataset(InMemoryDataset):
    """
    增强的PyTorch Geometric数据集，支持几何特征
    Enhanced PyTorch Geometric dataset with geometric features support
    """

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):
        # 如果没有指定root，使用默认的数据目录
        if root is None:
            root = str(path_manager.data_dir)
        
        self.db_path = path_manager.database_path
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # 指向原始数据库文件
        return [path_manager.database_path.name]

    @property
    def raw_dir(self):
        # 明确指定原始文件目录
        return str(path_manager.data_raw_dir)

    @property
    def processed_file_names(self):
        return ['pyg_patch_dataset_with_geometry_v2.pt'] # 使用新版本文件名

    def download(self):
        # 不需要下载，因为文件是本地生成的
        pass

    def process(self):
        """
        处理数据库，构建包含几何特征的PyG数据对象
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库未找到，请先运行populate_db.py: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # **CRITICAL FIX**: 从数据库中选择新添加的字段
        cursor.execute("""
                       SELECT id, edgebreaker_encoding, canonical_form, sides,
                              complexity_score, num_vertices, num_faces, quality,
                              boundary_vertices, vertex_normals, mean_curvatures,
                              gaussian_curvatures, edge_lengths, edge_curvatures,
                              avg_curvature, curvature_variance, total_boundary_length, area,
                              ordered_boundary_vertex_indices
                       FROM patterns
                       """)
        all_patterns = cursor.fetchall()
        conn.close()

        data_list = []
        quality_map = {'new': 1, 'old': 0}

        for row in tqdm.tqdm(all_patterns, desc="Processing Patches"):
            try:
                (id, edgebreaker_encoding, canonical_form, sides,
                 complexity_score, num_vertices, num_faces, quality,
                 boundary_vertices_json, vertex_normals_json, mean_curvatures_json,
                 gaussian_curvatures_json, edge_lengths_json, edge_curvatures_json,
                 avg_curvature, curvature_variance, total_boundary_length, area,
                 ordered_indices_json) = row

                parser = ProperPatternParser(pattern_string=edgebreaker_encoding, sides=sides)
                graph_data = parser.parse()
                
                # 验证graph_data的格式
                if not isinstance(graph_data, dict) or "num_nodes" not in graph_data or "edge_index" not in graph_data:
                    print(f"跳过样本 {id}: 解析结果格式错误")
                    continue
                    
                num_nodes = graph_data["num_nodes"]
                num_edges = graph_data["edge_index"].shape[1]
                
                # 基本验证
                if num_nodes <= 0 or num_edges < 0:
                    print(f"跳过样本 {id}: 无效的图结构 (nodes={num_nodes}, edges={num_edges})")
                    continue
            
            except Exception as e:
                print(f"跳过样本 {id}: 处理失败 - {e}")
                continue

            has_geometry = boundary_vertices_json is not None and ordered_indices_json is not None
            geometry_features = None
            ordered_boundary_indices = None
            if has_geometry:
                try:
                    geometry_features = {
                        'boundary_vertices': torch.tensor(json.loads(boundary_vertices_json), dtype=torch.float),
                        'vertex_normals': torch.tensor(json.loads(vertex_normals_json), dtype=torch.float),
                        'mean_curvatures': torch.tensor(json.loads(mean_curvatures_json), dtype=torch.float),
                        'gaussian_curvatures': torch.tensor(json.loads(gaussian_curvatures_json), dtype=torch.float),
                        'edge_lengths': torch.tensor(json.loads(edge_lengths_json), dtype=torch.float),
                        'edge_curvatures': torch.tensor(json.loads(edge_curvatures_json), dtype=torch.float),
                    }
                    ordered_boundary_indices = json.loads(ordered_indices_json)
                except (json.JSONDecodeError, TypeError):
                    has_geometry = False

            # --- 节点特征构建 (未修改) ---
            topology_features = torch.cat([
                graph_data["node_valence"].float().unsqueeze(1),
                graph_data["is_boundary_node"].float().unsqueeze(1),
                graph_data["is_corner_node"].float().unsqueeze(1),
                graph_data["distance_to_singular"].float().unsqueeze(1),
                graph_data["local_topology_config"].float().unsqueeze(1),
                graph_data["boundary_position_encoding"].float().unsqueeze(1)
            ], dim=1)
            
            # **CRITICAL FIX**: 修正节点几何特征的匹配
            # 原始的边界顶点（0到N-1）与几何特征是一一对应的
            x = torch.zeros(num_nodes, topology_features.shape[1] + 2)
            x[:, :topology_features.shape[1]] = topology_features

            if has_geometry and len(geometry_features['mean_curvatures']) == sides:
                # 假设几何特征是按照初始边界（0到sides-1）的顺序存储的
                for i in range(sides):
                    if i < num_nodes:
                        x[i, -2] = geometry_features['mean_curvatures'][i]
                        x[i, -1] = geometry_features['gaussian_curvatures'][i]

            # --- 边特征构建 (已修正) ---
            is_boundary_edge = graph_data["is_boundary_edge"].float().view(-1, 1)
            edge_length_features = torch.zeros(num_edges, 1)
            edge_curv_features = torch.zeros(num_edges, 1)

            if has_geometry and num_edges > 0:
                # **CRITICAL FIX**: 使用有序边界索引进行鲁棒的特征匹配
                
                # 1. 创建从局部顶点索引到全局顶点索引的映射
                # 假设解码器生成的顶点索引与编码器中的局部索引一致
                local_to_global_map = {i: ordered_boundary_indices[i] for i in range(sides) if i < len(ordered_boundary_indices)}
                
                # 2. 创建从全局边到其几何特征的映射
                global_edge_to_feature_map = {}
                num_geom_edges = len(geometry_features['edge_lengths'])
                for i in range(num_geom_edges):
                    global_v1 = ordered_boundary_indices[i]
                    global_v2 = ordered_boundary_indices[(i + 1) % num_geom_edges]
                    edge = tuple(sorted((global_v1, global_v2)))
                    global_edge_to_feature_map[edge] = (
                        geometry_features['edge_lengths'][i],
                        geometry_features['edge_curvatures'][i]
                    )

                # 3. 遍历图中的所有边，并使用映射分配特征
                edge_index_np = graph_data["edge_index"].t().numpy()
                for i in range(num_edges):
                    local_v1, local_v2 = edge_index_np[i]
                    
                    # 检查这条局部边是否是初始边界的一部分
                    if local_v1 in local_to_global_map and local_v2 in local_to_global_map:
                        global_v1 = local_to_global_map[local_v1]
                        global_v2 = local_to_global_map[local_v2]
                        
                        global_edge = tuple(sorted((global_v1, global_v2)))
                        
                        if global_edge in global_edge_to_feature_map:
                            length, curv = global_edge_to_feature_map[global_edge]
                            edge_length_features[i] = length
                            edge_curv_features[i] = curv

            edge_attr = torch.cat([is_boundary_edge, edge_length_features, edge_curv_features], dim=1)

            data = Data(
                x=x,
                edge_index=graph_data["edge_index"],
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                pattern_id=id,
                num_sides=sides,
                quality=quality_map.get(quality, 0),
                canonical_form=canonical_form,
                complexity_score=float(complexity_score or 0.0),
                has_geometry=has_geometry
            )
            data_list.append(data)

        if self.pre_filter:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
