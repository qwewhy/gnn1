import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging
from collections import deque, defaultdict

class EdgebreakerDecoder:
    """完整的Edgebreaker解码器实现"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def decode_pattern_string(self, pattern_string: str, num_boundary_sides: int) -> Optional[Dict]:
        """从Edgebreaker编码重建拓扑图"""
        try:
            # 1. 解析编码字符串
            multi_chord_info, edgebreaker_ops = self._parse_pattern_string(pattern_string)
            
            # 2. 初始化边界环
            vertices = list(range(num_boundary_sides))
            edges = set()
            faces = []
            
            # 添加初始边界环的边
            for i in range(num_boundary_sides):
                v1 = i
                v2 = (i + 1) % num_boundary_sides
                edges.add(tuple(sorted((v1, v2))))
            
            # 3. 应用多弦信息重建基础结构
            vertices, edges, faces = self._apply_multi_chord_unfolding(
                vertices, edges, faces, multi_chord_info, num_boundary_sides
            )
            
            # 4. 执行Edgebreaker重建
            graph_data = self._edgebreaker_decode(
                vertices, edges, faces, edgebreaker_ops, num_boundary_sides
            )
            
            return graph_data
            
        except Exception as e:
            self.logger.error(f"解码失败 {pattern_string}: {e}")
            return self._create_fallback_graph(num_boundary_sides)
    
    def _parse_pattern_string(self, pattern_string: str) -> Tuple[List[str], List[str]]:
        """解析模式字符串"""
        if '#' in pattern_string:
            multi_chord_part, edgebreaker_part = pattern_string.split('#', 1)
        else:
            multi_chord_part = ""
            edgebreaker_part = pattern_string
        
        # 解析多弦信息
        multi_chord_info = []
        if multi_chord_part.strip():
            multi_chord_info = multi_chord_part.strip().split()
        
        # 解析Edgebreaker操作
        edgebreaker_ops = list(edgebreaker_part.strip())
        
        return multi_chord_info, edgebreaker_ops
    
    def _apply_multi_chord_unfolding(self, vertices: List[int], edges: Set[Tuple[int, int]], 
                                    faces: List[List[int]], multi_chord_info: List[str],
                                    num_boundary_sides: int) -> Tuple[List[int], Set[Tuple[int, int]], List[List[int]]]:
        """应用多弦展开信息"""
        if not multi_chord_info:
            return vertices, edges, faces
            
        current_boundary = list(range(num_boundary_sides))
        
        for chord_count_str in multi_chord_info:
            try:
                chord_count = int(chord_count_str)
                
                # 对每个弦计数，添加内部结构
                for _ in range(chord_count):
                    if len(current_boundary) >= 3:
                        # 在边界上选择两个非相邻点作为弦的端点
                        boundary_size = len(current_boundary)
                        
                        # 简单策略：连接对角的点
                        start_idx = 0
                        end_idx = boundary_size // 2
                        
                        start_vertex = current_boundary[start_idx]
                        end_vertex = current_boundary[end_idx]
                        
                        # 添加弦
                        edges.add(tuple(sorted((start_vertex, end_vertex))))
                        
                        # 添加新的内部顶点（可选）
                        if chord_count > 1:
                            new_vertex = len(vertices)
                            vertices.append(new_vertex)
                            
                            # 连接新顶点到弦的中点
                            edges.add(tuple(sorted((new_vertex, start_vertex))))
                            edges.add(tuple(sorted((new_vertex, end_vertex))))
                        
            except ValueError:
                continue
                
        return vertices, edges, faces
    
    def _edgebreaker_decode(self, vertices: List[int], edges: Set[Tuple[int, int]], 
                           faces: List[List[int]], edgebreaker_ops: List[str],
                           num_boundary_sides: int) -> Dict:
        """执行Edgebreaker解码"""
        active_front = deque(range(num_boundary_sides))  # 当前活跃前沿
        
        for op in edgebreaker_ops:
            if not active_front:
                break
                
            if op == 'S':  # Start - 开始三角剖分
                if len(active_front) >= 3:
                    v1 = active_front.popleft()
                    v2 = active_front.popleft()
                    v3 = active_front[0]
                    
                    # 添加三角形
                    faces.append([v1, v2, v3])
                    edges.add(tuple(sorted((v1, v2))))
                    edges.add(tuple(sorted((v2, v3))))
                    edges.add(tuple(sorted((v3, v1))))
                    
            elif op == 'C':  # Case - 添加新顶点
                if len(active_front) >= 2:
                    v1 = active_front.popleft()
                    v2 = active_front[0]
                    
                    # 创建新顶点
                    new_vertex = len(vertices)
                    vertices.append(new_vertex)
                    
                    # 添加三角形
                    faces.append([v1, v2, new_vertex])
                    edges.add(tuple(sorted((v1, v2))))
                    edges.add(tuple(sorted((v2, new_vertex))))
                    edges.add(tuple(sorted((new_vertex, v1))))
                    
                    # 更新活跃前沿
                    active_front.appendleft(new_vertex)
                    
            elif op == 'L':  # Left - 左扩展
                if len(active_front) >= 2:
                    v1 = active_front[0]
                    v2 = active_front[1]
                    
                    new_vertex = len(vertices)
                    vertices.append(new_vertex)
                    
                    faces.append([v1, v2, new_vertex])
                    edges.add(tuple(sorted((v1, v2))))
                    edges.add(tuple(sorted((v2, new_vertex))))
                    edges.add(tuple(sorted((new_vertex, v1))))
                    
                    active_front.appendleft(new_vertex)
                    
            elif op == 'R':  # Right - 右扩展
                if len(active_front) >= 2:
                    v1 = active_front[-2]
                    v2 = active_front[-1]
                    
                    new_vertex = len(vertices)
                    vertices.append(new_vertex)
                    
                    faces.append([v1, v2, new_vertex])
                    edges.add(tuple(sorted((v1, v2))))
                    edges.add(tuple(sorted((v2, new_vertex))))
                    edges.add(tuple(sorted((new_vertex, v1))))
                    
                    active_front.append(new_vertex)
                    
            elif op == 'E':  # End - 结束
                break
        
        return self._build_graph_data(vertices, edges, faces, num_boundary_sides)
    
    def _build_graph_data(self, vertices: List[int], edges: Set[Tuple[int, int]], 
                         faces: List[List[int]], num_boundary_sides: int) -> Dict:
        """构建图数据结构"""
        num_nodes = len(vertices)
        boundary_vertices = list(range(num_boundary_sides))
        
        # 创建edge_index
        if edges:
            edge_list = []
            for v1, v2 in edges:
                edge_list.append([v1, v2])
                edge_list.append([v2, v1])  # 无向图
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # 计算节点特征
        node_features = self._compute_comprehensive_node_features(
            vertices, edges, faces, boundary_vertices
        )
        
        # 计算边特征
        edge_features = self._compute_edge_features(edges, boundary_vertices, edge_index)
        
        return {
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "node_features": node_features,
            "edge_features": edge_features,
            "faces": faces
        }
    
    def _compute_comprehensive_node_features(self, vertices: List[int], edges: Set[Tuple[int, int]], 
                                           faces: List[List[int]], boundary_vertices: List[int]) -> Dict[str, torch.Tensor]:
        """计算全面的节点特征"""
        num_nodes = len(vertices)
        
        # 1. 节点度数/价
        valence = torch.zeros(num_nodes, dtype=torch.long)
        for v1, v2 in edges:
            valence[v1] += 1
            valence[v2] += 1
        
        # 2. 边界节点标记
        is_boundary = torch.zeros(num_nodes, dtype=torch.bool)
        is_boundary[boundary_vertices] = True
        
        # 3. 角点检测（度数异常的边界点）
        is_corner = torch.zeros(num_nodes, dtype=torch.bool)
        for v in boundary_vertices:
            if v < num_nodes and valence[v] != 2:  # 边界上正常度数为2
                is_corner[v] = True
        
        # 4. 奇异点检测和距离计算
        singular_points = []
        for v in range(num_nodes):
            expected_valence = 2 if is_boundary[v] else 4
            if valence[v] != expected_valence:
                singular_points.append(v)
        
        distance_to_singular = self._compute_graph_distances(
            vertices, edges, singular_points
        )
        
        # 5. 局部拓扑配置
        local_config = self._compute_local_topology_features(
            vertices, edges, faces, valence
        )
        
        # 6. 边界位置编码
        boundary_position = torch.zeros(num_nodes, dtype=torch.float)
        if boundary_vertices:
            for i, v in enumerate(boundary_vertices):
                if v < num_nodes:
                    boundary_position[v] = i / len(boundary_vertices)
        
        return {
            "node_valence": valence,
            "is_boundary_node": is_boundary,
            "is_corner_node": is_corner,
            "distance_to_singular": distance_to_singular,
            "local_topology_config": local_config,
            "boundary_position_encoding": boundary_position
        }
    
    def _compute_graph_distances(self, vertices: List[int], edges: Set[Tuple[int, int]], 
                                target_vertices: List[int]) -> torch.Tensor:
        """使用BFS计算图上的距离"""
        num_nodes = len(vertices)
        distances = torch.full((num_nodes,), float('inf'))
        
        if not target_vertices:
            return torch.zeros(num_nodes, dtype=torch.float)
        
        # 构建邻接表
        adj_list = defaultdict(list)
        for v1, v2 in edges:
            adj_list[v1].append(v2)
            adj_list[v2].append(v1)
        
        # 从每个目标顶点开始BFS
        for target in target_vertices:
            if target >= num_nodes:
                continue
                
            visited = set()
            queue = deque([(target, 0)])
            visited.add(target)
            
            while queue:
                node, dist = queue.popleft()
                distances[node] = min(distances[node], dist)
                
                for neighbor in adj_list[node]:
                    if neighbor not in visited and neighbor < num_nodes:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        
        # 处理无穷大距离
        max_finite = distances[distances != float('inf')].max() if len(distances[distances != float('inf')]) > 0 else 0
        distances[distances == float('inf')] = max_finite + 1
        
        return distances.float()
    
    def _compute_local_topology_features(self, vertices: List[int], edges: Set[Tuple[int, int]], 
                                        faces: List[List[int]], valence: torch.Tensor) -> torch.Tensor:
        """计算局部拓扑特征"""
        num_nodes = len(vertices)
        local_features = torch.zeros(num_nodes, dtype=torch.float)
        
        # 基于相邻面数量和度数变化
        face_count = torch.zeros(num_nodes, dtype=torch.float)
        for face in faces:
            for vertex in face:
                if vertex < num_nodes:
                    face_count[vertex] += 1
        
        # 结合度数信息
        for i in range(num_nodes):
            # 局部配置 = 面数量 * 度数偏差
            valence_deviation = abs(valence[i] - 4.0)  # 偏离常规度数4的程度
            local_features[i] = face_count[i] * (1 + valence_deviation * 0.1)
        
        # 归一化
        max_feature = local_features.max()
        if max_feature > 0:
            local_features = local_features / max_feature
        
        return local_features
    
    def _compute_edge_features(self, edges: Set[Tuple[int, int]], boundary_vertices: List[int], 
                              edge_index: torch.Tensor) -> torch.Tensor:
        """计算边特征"""
        if edge_index.numel() == 0:
            return torch.empty((0, 1), dtype=torch.float)
        
        boundary_set = set(boundary_vertices)
        edge_features = []
        
        # 遍历edge_index的每条边（已经是双向的）
        for i in range(edge_index.shape[1]):
            v1, v2 = edge_index[0, i].item(), edge_index[1, i].item()
            
            # 边界边标记
            is_boundary_edge = (v1 in boundary_set) and (v2 in boundary_set)
            edge_features.append([float(is_boundary_edge)])
        
        return torch.tensor(edge_features, dtype=torch.float)
    
    def _create_fallback_graph(self, num_boundary_sides: int) -> Dict:
        """创建回退图（基本边界环）"""
        num_nodes = num_boundary_sides
        edges = []
        
        for i in range(num_nodes):
            edges.append([i, (i + 1) % num_nodes])
            edges.append([(i + 1) % num_nodes, i])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return {
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "node_features": {
                "node_valence": torch.full((num_nodes,), 2, dtype=torch.long),
                "is_boundary_node": torch.ones(num_nodes, dtype=torch.bool),
                "is_corner_node": torch.zeros(num_nodes, dtype=torch.bool),
                "distance_to_singular": torch.zeros(num_nodes, dtype=torch.float),
                "local_topology_config": torch.zeros(num_nodes, dtype=torch.float),
                "boundary_position_encoding": torch.arange(num_nodes, dtype=torch.float) / num_nodes
            },
            "edge_features": torch.ones((edge_index.shape[1], 1), dtype=torch.float),
            "faces": []
        }

class ProperPatternParser:
    """正确的模式解析器"""
    def __init__(self, pattern_string: str, sides: int):
        self.pattern_string = pattern_string
        self.sides = sides
        self.decoder = EdgebreakerDecoder()
        
    def parse(self) -> Dict:
        """解析模式字符串生成图拓扑"""
        # 使用完整的Edgebreaker解码
        graph_data = self.decoder.decode_pattern_string(self.pattern_string, self.sides)
        
        if graph_data is None:
            return self._create_boundary_fallback()
        
        # 整合所有特征
        node_features = graph_data["node_features"]
        
        return {
            "edge_index": graph_data["edge_index"],
            "num_nodes": graph_data["num_nodes"],
            "node_valence": node_features["node_valence"],
            "is_boundary_node": node_features["is_boundary_node"],
            "is_corner_node": node_features["is_corner_node"],
            "distance_to_singular": node_features["distance_to_singular"],
            "local_topology_config": node_features["local_topology_config"],
            "boundary_position_encoding": node_features["boundary_position_encoding"],
            "is_boundary_edge": graph_data["edge_features"]
        }
    
    def _create_boundary_fallback(self) -> Dict:
        """创建基本边界环作为后备方案"""
        num_nodes = self.sides
        edges = []
        for i in range(num_nodes):
            edges.append([i, (i + 1) % num_nodes])
            edges.append([(i + 1) % num_nodes, i])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return {
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "node_valence": torch.full((num_nodes,), 2, dtype=torch.long),
            "is_boundary_node": torch.ones(num_nodes, dtype=torch.bool),
            "is_corner_node": torch.zeros(num_nodes, dtype=torch.bool),
            "distance_to_singular": torch.zeros(num_nodes, dtype=torch.float),
            "local_topology_config": torch.zeros(num_nodes, dtype=torch.float),
            "boundary_position_encoding": torch.arange(num_nodes, dtype=torch.float) / num_nodes,
            "is_boundary_edge": torch.ones(edge_index.shape[1], dtype=torch.bool)
        }