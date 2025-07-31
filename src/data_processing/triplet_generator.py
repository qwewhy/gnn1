import trimesh
import numpy as np
import networkx as nx
from typing import Optional, List, Tuple
import torch
from torch_geometric.data import Data

from src.data_processing.pyg_dataset import PatchDataset
from src.data_processing.proper_encoder import ProperPatternEncoder


class TripletGenerator:
    """
    Generates (anchor, positive, negative) triplets for metric learning.
    为度量学习生成（锚点、正样本、负样本）三元组。

    This class loads a 3D mesh, extracts a patch from it, and uses that
    patch to create an "anchor" and a "positive" sample. A "negative"
    sample is then chosen from the general dataset.
    该类加载一个3D网格，从中提取一个面片，并使用该面片创建一个“锚点”
    和一个“正样本”。然后从通用数据集中选择一个“负样本”。
    """

    def __init__(self, mesh_path: str, patch_dataset: PatchDataset):
        """
        Initializes the generator with a source mesh and a patch dataset.
        使用源网格和面片数据集初始化生成器。

        Args:
            mesh_path (str): Path to the 3D model file (e.g., .obj).
                             3D模型文件的路径（例如.obj）。
            patch_dataset (PatchDataset): The dataset of all known topological patterns.
                                          包含所有已知拓扑模式的数据集。
        """
        try:
            self.mesh = trimesh.load(mesh_path, process=True)
            # Ensure we are working with a triangular mesh for simplicity
            # 为简单起见，确保我们处理的是三角网格
            self.mesh.merge_vertices()
            self.mesh.remove_degenerate_faces()
            self.mesh.remove_duplicate_faces()

            # Pre-calculate vertex curvatures for the entire mesh
            # 为整个网格预先计算顶点曲率
            self.mesh.vertex_curvatures = trimesh.curvature.discrete_mean_curvature_measure(
                self.mesh, self.mesh.vertices, radius=self.mesh.scale / 100.0
            )

            # We are interested in quad patches, so let's make sure the mesh has quads
            # 我们对面片感兴趣，所以要确保网格有四边形
            if not np.all(self.mesh.faces.shape == (self.mesh.faces.shape[0], 4)):
                 # If not all faces are quads, we can't proceed with quad logic.
                 # This is a simplification; a more robust approach could handle mixed meshes.
                 # 如果不全是四边形，我们就无法处理。这是一个简化；更鲁棒的方法可以处理混合网格。
                 pass # For now, we assume quad meshes are provided.
        except Exception as e:
            raise IOError(f"Failed to load or process mesh at {mesh_path}: {e}")

        self.patch_dataset = patch_dataset
        self.face_adjacency_graph = nx.from_edgelist(self.mesh.face_adjacency)
        self.encoder = ProperPatternEncoder()  # 用于真正的拓扑编码

    def _create_anchor_from_patch(self, patch_face_indices: List[int]) -> Optional[Data]:
        """
        Creates a PyG Data object for the anchor graph from a mesh patch.
        从网格面片为锚点图创建一个PyG Data对象。

        This involves finding the boundary loop of the patch and extracting
        geometric features for each vertex on the boundary.
        这涉及到找到面片的边界环，并为边界上的每个顶点提取几何特征。

        Args:
            patch_face_indices: A list of face indices forming the patch.
                                构成面片的面索引列表。

        Returns:
            A PyG Data object for the anchor, or None if creation fails.
            一个锚点的PyG Data对象，如果创建失败则返回None。
        """
        if not patch_face_indices:
            return None

        # Use trimesh to find the boundary edges of the patch
        # 使用trimesh找到面片的边界边
        boundary_path = self.mesh.outline(patch_face_indices)

        if boundary_path is None or len(boundary_path.vertices) == 0:
            return None

        # Extract boundary vertices from the path
        # 从路径中提取边界顶点
        boundary_vertex_indices = boundary_path.vertices
        
        # --- Feature Extraction ---
        # --- 特征提取 ---
        
        # 1. Pos, Normal, Curvature
        positions = self.mesh.vertices[boundary_vertex_indices]
        normals = self.mesh.vertex_normals[boundary_vertex_indices]
        curvatures = self.mesh.vertex_curvatures[boundary_vertex_indices]
        
        # 2. Boundary segment length ratio
        # 2. 边界段长度比
        boundary_points = torch.tensor(positions, dtype=torch.float)
        # Calculate edge lengths for the loop: ||p_i - p_{i-1}||
        segment_lengths = torch.norm(boundary_points - torch.roll(boundary_points, 1, dims=0), dim=1)
        total_boundary_length = torch.sum(segment_lengths)
        
        if total_boundary_length < 1e-6: # Avoid division by zero
            return None
        
        # Length for a vertex is the sum of its two adjacent segments
        # 一个顶点的长度是其两个相邻段的长度之和
        vertex_segment_lengths = segment_lengths + torch.roll(segment_lengths, -1, dims=0)
        length_ratios = vertex_segment_lengths / total_boundary_length

        # === 计算拓扑特征以匹配pattern特征 ===
        # === Compute topological features to match pattern features ===
        
        # 4. 节点度数 (在边界环中都是2)
        # 4. Node valence (all are 2 in boundary loop)
        valence = torch.full((len(boundary_vertex_indices),), 2.0)
        
        # 5. 边界节点标记 (都是True)
        # 5. Boundary node marker (all True)
        is_boundary_node = torch.ones(len(boundary_vertex_indices))
        
        # 6. 角点检测 (基于曲率)
        # 6. Corner detection (based on curvature)
        curvature_threshold = torch.quantile(torch.abs(torch.tensor(curvatures)), 0.8)
        is_corner_node = (torch.abs(torch.tensor(curvatures)) > curvature_threshold).float()
        
        # 7. 到高曲率点的距离 (模拟到奇异点的距离)
        # 7. Distance to high curvature points (simulate distance to singular points)
        distance_to_singular = self._compute_distance_to_high_curvature(
            torch.tensor(curvatures), len(boundary_vertex_indices)
        )
        
        # 8. 局部几何配置 (基于曲率变化)
        # 8. Local geometric configuration (based on curvature variation)
        local_config = self._compute_local_geometric_config(torch.tensor(curvatures))
        
        # 9. 边界位置编码 (与pattern一致)
        # 9. Boundary position encoding (consistent with pattern)
        boundary_position_encoding = torch.arange(len(boundary_vertex_indices), dtype=torch.float) / len(boundary_vertex_indices)
        
        # 将几何和拓扑特征结合 (6维拓扑 + 2维几何 = 8维)
        # Combine geometric and topological features (6D topological + 2D geometric = 8D)
        x = torch.cat([
            # 拓扑特征 (6维) - 与pattern特征对应
            # Topological features (6D) - corresponding to pattern features
            valence.unsqueeze(1),                    # (1D) 度数
            is_boundary_node.unsqueeze(1),           # (1D) 边界标记  
            is_corner_node.unsqueeze(1),             # (1D) 角点标记
            distance_to_singular.unsqueeze(1),       # (1D) 到奇异点距离
            local_config.unsqueeze(1),               # (1D) 局部配置
            boundary_position_encoding.unsqueeze(1), # (1D) 边界位置
            
            # 额外几何特征 (2维) - anchor特有
            # Additional geometric features (2D) - anchor specific
            torch.tensor(curvatures, dtype=torch.float).unsqueeze(1),  # (1D) 曲率
            length_ratios.unsqueeze(1)                                  # (1D) 长度比
        ], dim=1)  # 总共8维特征

        # Create edge_index for the boundary loop
        # 为边界环创建 edge_index
        num_boundary_nodes = len(boundary_vertex_indices)
        loop_indices = torch.arange(num_boundary_nodes, dtype=torch.long)
        edge_index = torch.stack([
            loop_indices,
            torch.roll(loop_indices, -1, dims=0)
        ], dim=0)
        
        # Create the Data object
        # 创建Data对象
        anchor_data = Data(
            x=x,
            edge_index=edge_index,
            num_nodes=num_boundary_nodes,
            num_sides=num_boundary_nodes
        )
        
        return anchor_data

    def extract_random_patch(self, min_faces: int = 10, max_faces: int = 20) -> Optional[List[int]]:
        """
        Extracts a patch of connected faces from the mesh using BFS.
        使用BFS从网格中提取一个由相连面组成的面片。

        Args:
            min_faces (int): The minimum number of faces in the patch.
                             面片中最小的面数。
            max_faces (int): The maximum number of faces in the patch.
                             面片中最大的面数。

        Returns:
            A list of face indices representing the patch, or None if extraction fails.
            代表面片的的面索引列表，如果提取失败则返回None。
        """
        num_total_faces = len(self.mesh.faces)
        if num_total_faces == 0:
            return None

        # Try a few times to find a suitable patch
        # 尝试几次以找到合适的面片
        for _ in range(10): # 10 attempts
            start_face_idx = np.random.randint(0, num_total_faces)
            
            # Use BFS to find a patch of faces
            # 使用BFS寻找面片
            q = [start_face_idx]
            visited = {start_face_idx}
            patch_faces = [start_face_idx]
            
            while q:
                current_face = q.pop(0)
                
                # Stop if we have enough faces
                # 如果面数足够就停止
                if len(patch_faces) >= max_faces:
                    break
                
                # Find neighbors and add them to the queue
                # 寻找邻居并加入队列
                for neighbor in self.face_adjacency_graph.neighbors(current_face):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        patch_faces.append(neighbor)
                        q.append(neighbor)
                        if len(patch_faces) >= max_faces:
                            break
            
            if min_faces <= len(patch_faces) <= max_faces:
                return patch_faces
        
        return None # Failed to find a suitable patch

    def generate_triplet(self) -> Optional[Tuple[Data, Data, Data]]:
        """
        Generates a single (anchor, positive, negative) triplet.
        生成一个（锚点，正样本，负样本）三元组。

        Returns:
            A tuple containing the anchor, positive, and negative data samples,
            or None if generation fails.
            一个包含锚点、正样本和负样本数据的元组，如果生成失败则返回None。
        """
        # 1. Extract a geometric patch from the mesh
        # 1. 从网格中提取一个几何面片
        patch_face_indices = self.extract_random_patch()
        if not patch_face_indices:
            print("Warning: Could not extract a suitable patch from the mesh.")
            return None

        # 2. Create Anchor (Query Graph with Geometry) by processing the patch
        # 2. 通过处理面片创建锚点（带几何信息的查询图）
        anchor = self._create_anchor_from_patch(patch_face_indices)
        if anchor is None:
            print("Warning: Failed to create anchor from the extracted patch.")
            return None

        # 3. Create Positive (Topological Pattern Graph) using proper encoding
        # 3. 使用正确编码创建正样本（拓扑模式图）
        
        # 使用真正的拓扑编码找到匹配的模式
        positive_idx = self._find_topological_match(patch_face_indices, anchor.num_sides)
        
        if positive_idx is None:
            print(f"Warning: No topological match found for patch with {anchor.num_sides} sides.")
            return None
            
        positive = self.patch_dataset[positive_idx]

        # 4. Create Negative (Different Topological Pattern)
        # 4. 创建负样本（不同的拓扑模式图）
        
        # 找到与正样本拓扑不同的负样本
        negative_idx = self._find_topological_negative(positive_idx, anchor.num_sides)
        
        if negative_idx is None:
            print(f"Warning: No suitable negative found for anchor with {anchor.num_sides} sides.")
            return None
            
        negative = self.patch_dataset[negative_idx]
        
        return anchor, positive, negative
    
    def _find_topological_match(self, patch_face_indices: List[int], num_sides: int) -> Optional[int]:
        """
        使用真正的拓扑编码找到匹配的模式
        Find matching pattern using proper topological encoding
        """
        try:
            # 1. 编码当前面片
            encoding_result = self.encoder.encode_patch_to_pattern(self.mesh, patch_face_indices)
            if encoding_result is None:
                return None
                
            edgebreaker_encoding, _ = encoding_result
            
            # 2. 标准化编码以便匹配
            canonical_form = self._normalize_encoding_for_matching(edgebreaker_encoding)
            
            # 3. 在数据集中寻找相同的规范形式
            for i, data in enumerate(self.patch_dataset):
                if (data.num_sides == num_sides and 
                    hasattr(data, 'canonical_form') and
                    self._encodings_match(data.canonical_form, canonical_form)):
                    # 优先选择 'new' 质量的样本
                    if data.quality == 1:
                        return i
            
            # 如果没有找到完全匹配的 'new' 样本，再找 'old' 样本
            for i, data in enumerate(self.patch_dataset):
                if (data.num_sides == num_sides and 
                    hasattr(data, 'canonical_form') and
                    self._encodings_match(data.canonical_form, canonical_form)):
                    return i
            
            # 如果没有找到拓扑匹配，回退到边数匹配+质量优先
            print("Warning: No exact topological match found, falling back to side-count matching")
            return self._fallback_match_by_sides(num_sides)
            
        except Exception as e:
            print(f"Error in topological matching: {e}")
            return self._fallback_match_by_sides(num_sides)
    
    def _find_topological_negative(self, positive_idx: int, num_sides: int) -> Optional[int]:
        """
        找到与正样本拓扑不同的负样本
        Find negative sample with different topology from positive
        """
        positive_data = self.patch_dataset[positive_idx]
        positive_canonical = getattr(positive_data, 'canonical_form', None)
        
        # 收集所有可能的负样本（相同边数但不同拓扑）
        possible_negatives = []
        
        for i, data in enumerate(self.patch_dataset):
            if (i != positive_idx and 
                data.num_sides == num_sides and
                not self._encodings_match(
                    getattr(data, 'canonical_form', ''), 
                    positive_canonical or ''
                )):
                possible_negatives.append(i)
        
        if not possible_negatives:
            # 如果找不到不同拓扑的样本，从不同边数中选择
            print("Warning: No different topology found for same side count, using different side count")
            for i, data in enumerate(self.patch_dataset):
                if i != positive_idx and data.num_sides != num_sides:
                    possible_negatives.append(i)
        
        if possible_negatives:
            return np.random.choice(possible_negatives)
        
        return None
    
    def _normalize_encoding_for_matching(self, encoding: str) -> str:
        """标准化编码用于匹配"""
        # 简化实现：移除空格并转换为小写
        return encoding.strip().lower()
    
    def _encodings_match(self, encoding1: str, encoding2: str) -> bool:
        """检查两个编码是否表示相同拓扑"""
        if not encoding1 or not encoding2:
            return False
        
        norm1 = self._normalize_encoding_for_matching(encoding1)
        norm2 = self._normalize_encoding_for_matching(encoding2)
        
        # 基本匹配：字符串相等
        if norm1 == norm2:
            return True
        
        # 在真正的实现中，这里会检查：
        # 1. 旋转对称性
        # 2. 镜像对称性
        # 3. 拓扑等价性
        
        return False
    
    def _fallback_match_by_sides(self, num_sides: int) -> Optional[int]:
        """回退方案：按边数匹配，优先选择'new'质量"""
        # 优先选择 'new' 质量的样本
        possible_positives_new = [
            i for i, data in enumerate(self.patch_dataset)
            if data.num_sides == num_sides and data.quality == 1
        ]
        
        if possible_positives_new:
            return np.random.choice(possible_positives_new)
        
        # 回退到 'old' 样本
        possible_positives_old = [
            i for i, data in enumerate(self.patch_dataset)
            if data.num_sides == num_sides
        ]
        
        if possible_positives_old:
            return np.random.choice(possible_positives_old)
        
        return None
    
    def _compute_distance_to_high_curvature(self, curvatures: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        计算到高曲率点的距离 (模拟到奇异点的距离)
        Compute distance to high curvature points (simulate distance to singular points)
        """
        # 找到高曲率点 (前20%的点)
        curvature_threshold = torch.quantile(torch.abs(curvatures), 0.8)
        high_curvature_indices = torch.where(torch.abs(curvatures) > curvature_threshold)[0]
        
        if len(high_curvature_indices) == 0:
            return torch.zeros(num_nodes)
            
        distances = torch.full((num_nodes,), float('inf'))
        
        # 对每个高曲率点计算到所有点的距离
        for singular_idx in high_curvature_indices:
            for i in range(num_nodes):
                # 在循环边界上的距离
                dist = min(abs(i - singular_idx), num_nodes - abs(i - singular_idx))
                distances[i] = min(distances[i], dist)
        
        # 归一化
        max_dist = distances[distances != float('inf')].max() if len(distances[distances != float('inf')]) > 0 else 1
        distances[distances == float('inf')] = max_dist
        
        return distances / max_dist if max_dist > 0 else distances
    
    def _compute_local_geometric_config(self, curvatures: torch.Tensor) -> torch.Tensor:
        """
        计算局部几何配置 (基于曲率变化)
        Compute local geometric configuration (based on curvature variation)
        """
        num_nodes = len(curvatures)
        local_config = torch.zeros(num_nodes)
        
        # 计算每个点周围的曲率变化
        for i in range(num_nodes):
            # 考虑相邻的点
            prev_idx = (i - 1) % num_nodes
            next_idx = (i + 1) % num_nodes
            
            # 计算曲率的局部变化
            curvature_variation = abs(curvatures[i] - curvatures[prev_idx]) + abs(curvatures[i] - curvatures[next_idx])
            local_config[i] = curvature_variation
        
        # 归一化
        max_variation = local_config.max()
        if max_variation > 0:
            local_config = local_config / max_variation
            
        return local_config 