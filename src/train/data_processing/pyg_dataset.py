# File: src/train/data_processing/pyg_dataset.py
# PyG数据集模块 / PyG dataset module

import sqlite3
import torch
from torch_geometric.data import InMemoryDataset, Data
from pathlib import Path
import tqdm
import json

# 使用集中的路径管理器
from src.common.path_manager import path_manager
from src.train.data_processing.proper_decoder import ProperPatternParser


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
        return ['pyg_patch_dataset_with_geometry.pt']

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

        cursor.execute("""
                       SELECT id,
                              edgebreaker_encoding,
                              canonical_form,
                              sides,
                              complexity_score,
                              num_vertices,
                              num_faces,
                              quality,
                              boundary_vertices,
                              vertex_normals,
                              mean_curvatures,
                              gaussian_curvatures,
                              edge_lengths,
                              edge_curvatures,
                              avg_curvature,
                              curvature_variance,
                              total_boundary_length,
                              area
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
                 avg_curvature, curvature_variance, total_boundary_length, area) = row

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

            has_geometry = boundary_vertices_json is not None
            geometry_features = None
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
                except (json.JSONDecodeError, TypeError):
                    has_geometry = False

            # 统一节点特征为8维
            topology_features = torch.cat([
                graph_data["node_valence"].float().unsqueeze(1),
                graph_data["is_boundary_node"].float().unsqueeze(1),
                graph_data["is_corner_node"].float().unsqueeze(1),
                graph_data["distance_to_singular"].float().unsqueeze(1),
                graph_data["local_topology_config"].float().unsqueeze(1),
                graph_data["boundary_position_encoding"].float().unsqueeze(1)
            ], dim=1)

            if has_geometry and num_nodes > 0 and num_nodes == len(geometry_features['mean_curvatures']):
                mean_curv = geometry_features['mean_curvatures'].unsqueeze(1)
                gauss_curv = geometry_features['gaussian_curvatures'].unsqueeze(1)
                x = torch.cat([topology_features, mean_curv, gauss_curv], dim=1)
            else:
                padding = torch.zeros(num_nodes, 2)
                x = torch.cat([topology_features, padding], dim=1)

            # 统一边特征为3维
            is_boundary_edge = graph_data["is_boundary_edge"].float().view(-1, 1)

            if has_geometry and num_edges > 0:
                edge_length_features = torch.zeros(num_edges, 1)
                edge_curv_features = torch.zeros(num_edges, 1)

                boundary_edge_mask = graph_data["is_boundary_edge"]
                if boundary_edge_mask.sum() > 0 and len(geometry_features['edge_lengths']) > 0:
                    boundary_edge_indices = torch.where(boundary_edge_mask)[0]
                    for i, edge_idx in enumerate(boundary_edge_indices):
                        geom_idx = i % len(geometry_features['edge_lengths'])
                        edge_length_features[edge_idx] = geometry_features['edge_lengths'][geom_idx]
                        edge_curv_features[edge_idx] = geometry_features['edge_curvatures'][geom_idx]

                edge_attr = torch.cat([is_boundary_edge, edge_length_features, edge_curv_features], dim=1)
            else:
                padding = torch.zeros(num_edges, 2)
                edge_attr = torch.cat([is_boundary_edge, padding], dim=1)

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
