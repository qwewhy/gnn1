import sqlite3
import torch
from torch_geometric.data import InMemoryDataset, Data
from pathlib import Path
import tqdm
import json

from src.data_processing.proper_decoder import ProperPatternParser


class PatchDataset(InMemoryDataset):
    """
    增强的PyTorch Geometric数据集，支持几何特征
    Enhanced PyTorch Geometric dataset with geometric features support
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.db_path = Path(root) / 'raw' / 'patches.db'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['patches.db']

    @property
    def processed_file_names(self):
        return ['pyg_patch_dataset_with_geometry.pt']  # 新文件名，避免与旧缓存冲突

    def download(self):
        pass

    def process(self):
        """
        处理数据库，构建包含几何特征的PyG数据对象
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 查询所有字段，包括几何特征
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

        for row in tqdm.tqdm(all_patterns, desc="处理面片"):
            (id, edgebreaker_encoding, canonical_form, sides,
             complexity_score, num_vertices, num_faces, quality,
             boundary_vertices_json, vertex_normals_json, mean_curvatures_json,
             gaussian_curvatures_json, edge_lengths_json, edge_curvatures_json,
             avg_curvature, curvature_variance, total_boundary_length, area) = row

            # 1. 解析拓扑结构
            parser = ProperPatternParser(pattern_string=edgebreaker_encoding, sides=sides)
            graph_data = parser.parse()

            # 2. 解析几何特征（如果存在）
            has_geometry = boundary_vertices_json is not None

            if has_geometry:
                try:
                    # 解析JSON格式的几何数据
                    boundary_vertices = json.loads(boundary_vertices_json)
                    vertex_normals = json.loads(vertex_normals_json)
                    mean_curvatures = json.loads(mean_curvatures_json)
                    gaussian_curvatures = json.loads(gaussian_curvatures_json)
                    edge_lengths = json.loads(edge_lengths_json)
                    edge_curvatures = json.loads(edge_curvatures_json)

                    # 将几何特征转换为张量
                    geometry_features = {
                        'boundary_vertices': torch.tensor(boundary_vertices, dtype=torch.float),
                        'vertex_normals': torch.tensor(vertex_normals, dtype=torch.float),
                        'mean_curvatures': torch.tensor(mean_curvatures, dtype=torch.float),
                        'gaussian_curvatures': torch.tensor(gaussian_curvatures, dtype=torch.float),
                        'edge_lengths': torch.tensor(edge_lengths, dtype=torch.float),
                        'edge_curvatures': torch.tensor(edge_curvatures, dtype=torch.float),
                    }
                except Exception as e:
                    print(f"解析几何特征失败 for pattern {id}: {e}")
                    has_geometry = False
                    geometry_features = None
            else:
                geometry_features = None

            # 3. 构建节点特征（拓扑特征 + 可选的几何特征）
            num_nodes = graph_data["num_nodes"]

            # 基础拓扑特征（6维）
            valence = graph_data["node_valence"].float().unsqueeze(1)
            is_boundary = graph_data["is_boundary_node"].float().unsqueeze(1)
            is_corner = graph_data["is_corner_node"].float().unsqueeze(1)
            distance_to_singular = graph_data["distance_to_singular"].float().unsqueeze(1)
            local_config = graph_data["local_topology_config"].float().unsqueeze(1)
            boundary_position = graph_data["boundary_position_encoding"].float().unsqueeze(1)

            topology_features = torch.cat([
                valence, is_boundary, is_corner,
                distance_to_singular, local_config, boundary_position
            ], dim=1)

            # 如果有几何特征，添加到节点特征中
            if has_geometry and num_nodes == len(geometry_features['mean_curvatures']):
                # 添加曲率特征（2维）
                mean_curv = geometry_features['mean_curvatures'].unsqueeze(1)
                gauss_curv = geometry_features['gaussian_curvatures'].unsqueeze(1)

                x = torch.cat([topology_features, mean_curv, gauss_curv], dim=1)  # 8维特征
            else:
                x = topology_features  # 仅6维拓扑特征

            # 4. 边特征
            edge_attr = graph_data["is_boundary_edge"].float().unsqueeze(1)

            # 如果有几何特征，为边添加长度和曲率
            if has_geometry and graph_data["edge_index"].shape[1] > 0:
                # 创建边到边界边的映射
                # 这里简化处理：假设边界边的几何特征可以映射到所有边
                num_edges = graph_data["edge_index"].shape[1]
                edge_length_features = torch.zeros(num_edges, 1)
                edge_curv_features = torch.zeros(num_edges, 1)

                # 为边界边赋值（简化实现）
                boundary_edge_mask = graph_data["is_boundary_edge"]
                if boundary_edge_mask.sum() > 0 and len(geometry_features['edge_lengths']) > 0:
                    # 循环使用边界边的几何特征
                    boundary_edge_indices = torch.where(boundary_edge_mask)[0]
                    for i, edge_idx in enumerate(boundary_edge_indices):
                        geom_idx = i % len(geometry_features['edge_lengths'])
                        edge_length_features[edge_idx] = geometry_features['edge_lengths'][geom_idx]
                        edge_curv_features[edge_idx] = geometry_features['edge_curvatures'][geom_idx]

                edge_attr = torch.cat([edge_attr, edge_length_features, edge_curv_features], dim=1)

            # 5. 创建PyG数据对象
            data = Data(
                x=x,
                edge_index=graph_data["edge_index"],
                edge_attr=edge_attr,
                num_nodes=num_nodes,
                # 元数据
                pattern_id=id,
                num_sides=sides,
                quality=quality_map.get(quality, 0),
                edgebreaker_encoding=edgebreaker_encoding,
                canonical_form=canonical_form,
                complexity_score=float(complexity_score or 0.0),
                num_vertices_orig=int(num_vertices or 0),
                num_faces_orig=int(num_faces or 0),
                # 几何统计信息
                has_geometry=has_geometry,
                avg_curvature=float(avg_curvature) if avg_curvature is not None else 0.0,
                curvature_variance=float(curvature_variance) if curvature_variance is not None else 0.0,
                total_boundary_length=float(total_boundary_length) if total_boundary_length is not None else 0.0,
                area=float(area) if area is not None else 0.0
            )

            # 如果有完整的几何数据，也存储原始数据供后续使用
            if geometry_features is not None:
                data.geometry_features = geometry_features

            data_list.append(data)

        # 应用过滤器和预变换
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        print(f"数据集创建完成：{len(data_list)} 个样本")
        print(f"特征维度：{data_list[0].x.shape[1] if data_list else 'N/A'}")
        print(f"包含几何特征的样本数：{sum(1 for d in data_list if d.has_geometry)}")