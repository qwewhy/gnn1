# File: src/train/data_processing/triplet_generator.py
# 三元组生成器 / Triplet generator

import trimesh
import numpy as np
import networkx as nx
from typing import Optional, List, Tuple
import torch
from torch_geometric.data import Data

from src.train.data_processing.pyg_dataset import PatchDataset
from src.train.data_processing.proper_encoder import ProperPatternEncoder


class TripletGenerator:
    """
    为度量学习生成（锚点、正样本、负样本）三元组。
    """

    def __init__(self, mesh_path: str, patch_dataset: PatchDataset):
        try:
            self.mesh = trimesh.load(mesh_path, process=True)
            self.mesh.merge_vertices()
            self.mesh.remove_degenerate_faces()
            self.mesh.remove_duplicate_faces()
            self.mesh.vertex_curvatures = trimesh.curvature.discrete_mean_curvature_measure(
                self.mesh, self.mesh.vertices, radius=self.mesh.scale / 100.0
            )
        except Exception as e:
            raise IOError(f"Failed to load or process mesh at {mesh_path}: {e}")

        self.patch_dataset = patch_dataset
        self.face_adjacency_graph = nx.from_edgelist(self.mesh.face_adjacency)
        self.encoder = ProperPatternEncoder()

    def _create_anchor_from_patch(self, patch_face_indices: List[int]) -> Optional[Data]:
        if not patch_face_indices:
            return None

        try:
            boundary_path = self.mesh.outline(patch_face_indices)
        except Exception:
            return None

        if not boundary_path or len(boundary_path.entities) != 1:
            return None

        boundary_vertex_indices = np.array(boundary_path.entities[0].points, dtype=np.int64)
        num_boundary_nodes = len(boundary_vertex_indices)
        if num_boundary_nodes == 0:
            return None

        # --- 节点特征 (Node Features) ---
        positions = self.mesh.vertices[boundary_vertex_indices]
        numpy_curvatures = self.mesh.vertex_curvatures[boundary_vertex_indices]
        curvatures_tensor = torch.tensor(numpy_curvatures, dtype=torch.float)
        if len(curvatures_tensor.shape) > 1:
            curvatures_tensor = torch.norm(curvatures_tensor, p=2, dim=1)
        curvatures_tensor = curvatures_tensor.squeeze()

        boundary_points = torch.tensor(positions, dtype=torch.float)
        segment_lengths = torch.norm(boundary_points - torch.roll(boundary_points, 1, dims=0), dim=1)
        total_boundary_length = torch.sum(segment_lengths)
        if total_boundary_length < 1e-6: return None
        vertex_segment_lengths = segment_lengths + torch.roll(segment_lengths, -1, dims=0)
        length_ratios = vertex_segment_lengths / total_boundary_length

        valence = torch.full((num_boundary_nodes,), 2.0)
        is_boundary_node = torch.ones(num_boundary_nodes)
        curvature_threshold = torch.quantile(torch.abs(curvatures_tensor), 0.8) if len(curvatures_tensor) > 0 else 0
        is_corner_node = (torch.abs(curvatures_tensor) > curvature_threshold).float()
        distance_to_singular = self._compute_distance_to_high_curvature(curvatures_tensor, num_boundary_nodes)
        local_config = self._compute_local_geometric_config(curvatures_tensor)
        boundary_position_encoding = torch.arange(num_boundary_nodes, dtype=torch.float) / num_boundary_nodes

        x = torch.stack([
            valence, is_boundary_node, is_corner_node, distance_to_singular,
            local_config, boundary_position_encoding, curvatures_tensor, length_ratios
        ], dim=1)

        # --- 边特征 (Edge Features) - 这是新增的关键修复 ---
        num_edges = num_boundary_nodes

        # 1. is_boundary_edge: 锚点全是边界边
        is_boundary_edge = torch.ones(num_edges, 1)

        # 2. edge_length: 每条边的长度
        edge_lengths = segment_lengths.view(-1, 1)

        # 3. edge_curvature: 每条边的曲率（用两端点曲率均值模拟）
        edge_curvatures = (curvatures_tensor + torch.roll(curvatures_tensor, -1, dims=0)) / 2
        edge_curvatures = edge_curvatures.view(-1, 1)

        # 拼接成3维边特征
        edge_attr = torch.cat([is_boundary_edge, edge_lengths, edge_curvatures], dim=1)

        # --- 创建图 (Create Graph) ---
        loop_indices = torch.arange(num_boundary_nodes, dtype=torch.long)
        edge_index = torch.stack([loop_indices, torch.roll(loop_indices, -1, dims=0)], dim=0)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,  # 添加边特征
            num_nodes=num_boundary_nodes,
            num_sides=num_boundary_nodes
        )

    def extract_random_patch(self, min_faces: int = 10, max_faces: int = 20) -> Optional[List[int]]:
        num_total_faces = len(self.mesh.faces)
        if num_total_faces < min_faces: return None
        for _ in range(10):
            start_face_idx = np.random.randint(0, num_total_faces)
            q, visited, patch_faces = [start_face_idx], {start_face_idx}, [start_face_idx]
            while q and len(patch_faces) < max_faces:
                current_face = q.pop(0)
                for neighbor in self.face_adjacency_graph.neighbors(current_face):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        patch_faces.append(neighbor)
                        q.append(neighbor)
            if min_faces <= len(patch_faces) <= max_faces:
                return patch_faces
        return None

    def generate_triplet(self) -> Optional[Tuple[Data, Data, Data]]:
        patch_face_indices = self.extract_random_patch()
        if not patch_face_indices: return None
        anchor = self._create_anchor_from_patch(patch_face_indices)
        if anchor is None: return None
        positive_idx = self._find_topological_match(patch_face_indices, anchor.num_sides)
        if positive_idx is None: return None
        positive = self.patch_dataset[positive_idx]
        negative_idx = self._find_topological_negative(positive_idx, anchor.num_sides)
        if negative_idx is None: return None
        negative = self.patch_dataset[negative_idx]
        return anchor, positive, negative

    def _find_topological_match(self, patch_face_indices: List[int], num_sides: int) -> Optional[int]:
        try:
            encoding_result = self.encoder.encode_patch_to_pattern(self.mesh, patch_face_indices)
            if not encoding_result: return self._fallback_match_by_sides(num_sides)
            canonical_form = self._normalize_encoding_for_matching(encoding_result[0])

            # 优先找高质量的完全匹配
            for i, data in enumerate(self.patch_dataset):
                if data.num_sides == num_sides and hasattr(data, 'canonical_form') and self._encodings_match(
                        data.canonical_form, canonical_form) and data.quality == 1:
                    return i
            # 再找普通质量的完全匹配
            for i, data in enumerate(self.patch_dataset):
                if data.num_sides == num_sides and hasattr(data, 'canonical_form') and self._encodings_match(
                        data.canonical_form, canonical_form):
                    return i
            return self._fallback_match_by_sides(num_sides)
        except Exception:
            return self._fallback_match_by_sides(num_sides)

    def _find_topological_negative(self, positive_idx: int, num_sides: int) -> Optional[int]:
        positive_data = self.patch_dataset[positive_idx]
        positive_canonical = getattr(positive_data, 'canonical_form', None)

        possible_negatives = [i for i, data in enumerate(self.patch_dataset) if
                              i != positive_idx and data.num_sides == num_sides and not self._encodings_match(
                                  getattr(data, 'canonical_form', ''), positive_canonical or '')]
        if not possible_negatives:
            possible_negatives = [i for i, data in enumerate(self.patch_dataset) if data.num_sides != num_sides]

        return np.random.choice(possible_negatives) if possible_negatives else None

    def _normalize_encoding_for_matching(self, encoding: str) -> str:
        return encoding.strip().lower()

    def _encodings_match(self, encoding1: str, encoding2: str) -> bool:
        if not encoding1 or not encoding2: return False
        return self._normalize_encoding_for_matching(encoding1) == self._normalize_encoding_for_matching(encoding2)

    def _fallback_match_by_sides(self, num_sides: int) -> Optional[int]:
        possible_positives = [i for i, data in enumerate(self.patch_dataset) if
                              data.num_sides == num_sides and data.quality == 1]
        if not possible_positives:
            possible_positives = [i for i, data in enumerate(self.patch_dataset) if data.num_sides == num_sides]
        return np.random.choice(possible_positives) if possible_positives else None

    def _compute_distance_to_high_curvature(self, curvatures: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if num_nodes == 0 or len(curvatures) == 0: return torch.zeros(num_nodes)

        abs_curvatures = torch.abs(curvatures)
        # Handle case where all curvatures are zero
        if torch.all(abs_curvatures == 0): return torch.zeros(num_nodes)

        curvature_threshold = torch.quantile(abs_curvatures, 0.8)
        high_curvature_indices = torch.where(abs_curvatures > curvature_threshold)[0]
        if len(high_curvature_indices) == 0: return torch.zeros(num_nodes)

        all_indices = torch.arange(num_nodes)
        # Use broadcasting for efficient distance calculation
        distances = torch.abs(all_indices.view(-1, 1) - high_curvature_indices)
        cyclic_distances = torch.min(distances, num_nodes - distances)
        min_distances, _ = torch.min(cyclic_distances, dim=1)

        # Normalize
        max_dist = min_distances.max()
        return min_distances / max_dist if max_dist > 0 else torch.zeros(num_nodes)

    def _compute_local_geometric_config(self, curvatures: torch.Tensor) -> torch.Tensor:
        num_nodes = len(curvatures)
        if num_nodes == 0: return torch.zeros(0)

        prev_curv = torch.roll(curvatures, 1, dims=0)
        next_curv = torch.roll(curvatures, -1, dims=0)

        curvature_variation = torch.abs(curvatures - prev_curv) + torch.abs(curvatures - next_curv)

        max_variation = curvature_variation.max()
        return curvature_variation / max_variation if max_variation > 0 else torch.zeros(num_nodes)