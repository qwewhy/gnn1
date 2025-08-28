# File: src/train/data_processing/proper_decoder.py
# 模式解码器 / Pattern decoder

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
            # 1. 解析编码字符串，直接处理EdgeBreaker操作
            edgebreaker_ops = self._parse_edgebreaker_string(pattern_string)
            
            # 2. 初始化边界环
            vertices = list(range(num_boundary_sides))
            edges = set()
            faces = []
            
            # 添加初始边界环的边
            for i in range(num_boundary_sides):
                v1 = i
                v2 = (i + 1) % num_boundary_sides
                edges.add(tuple(sorted((v1, v2))))
            
            # 3. 执行Edgebreaker重建
            graph_data = self._edgebreaker_decode(
                vertices, edges, faces, edgebreaker_ops, num_boundary_sides
            )
            
            return graph_data
            
        except Exception as e:
            self.logger.error(f"解码失败 {pattern_string}: {e}")
            return self._create_fallback_graph(num_boundary_sides)
    
    def _parse_edgebreaker_string(self, pattern_string: str) -> List[str]:
        """解析EdgeBreaker编码字符串"""
        # 处理EdgeBreaker操作
        if '#' in pattern_string:
            _, edgebreaker_part = pattern_string.split('#', 1)
        else:
            edgebreaker_part = pattern_string
        
        # 解析Edgebreaker操作
        edgebreaker_ops = list(edgebreaker_part.strip())
        
        return edgebreaker_ops
    

    
    def _edgebreaker_decode(self, vertices: List[int], edges: Set[Tuple[int, int]], 
                           faces: List[List[int]], edgebreaker_ops: List[str],
                           num_boundary_sides: int) -> Dict:
        """
        执行Edgebreaker解码 (修正的核心逻辑)
        Executes the Edgebreaker decoding (Corrected core logic).
        """
        if not edgebreaker_ops or num_boundary_sides < 3:
            return self._build_graph_data(vertices, edges, faces, num_boundary_sides)
            
        active_front = deque(range(num_boundary_sides))
        
        # 跳过可能存在的 'S' 操作符
        if edgebreaker_ops and edgebreaker_ops[0] == 'S':
            edgebreaker_ops = edgebreaker_ops[1:]

        for op in edgebreaker_ops:
            if len(active_front) < 2:
                break # 边界太小，无法继续操作
                
            # 门 (Gate) 是活动边界的前两个顶点
            v1, v2 = active_front[0], active_front[1]

            if op == 'C':  # Create: 添加新顶点
                # 创建一个新顶点
                new_vertex = len(vertices)
                vertices.append(new_vertex)
                
                # 形成新三角形 (v1, v2, new_vertex)
                faces.append([v1, v2, new_vertex])
                edges.add(tuple(sorted((v1, v2))))
                edges.add(tuple(sorted((v2, new_vertex))))
                edges.add(tuple(sorted((new_vertex, v1))))
                
                # 更新活动边界：用 (v1, new_vertex, v2) 替换 (v1, v2)
                # 这相当于在v1和v2之间插入new_vertex
                active_front[1] = new_vertex
                active_front.insert(2, v2)
                
            elif op == 'L':  # Left Zip: 与左侧顶点缝合
                if len(active_front) < 3: break
                # 第三个顶点是边界上v1的前一个顶点
                v3 = active_front[-1]
                
                # 形成新三角形 (v1, v2, v3)
                faces.append([v1, v2, v3])
                edges.add(tuple(sorted((v1, v2))))
                edges.add(tuple(sorted((v2, v3))))
                edges.add(tuple(sorted((v3, v1))))
                
                # 更新活动边界：v1成为内部点，将其移除
                active_front.popleft()

            elif op == 'R':  # Right Zip: 与右侧顶点缝合
                if len(active_front) < 3: break
                # 第三个顶点是边界上v2的后一个顶点
                v3 = active_front[2]

                # 形成新三角形 (v1, v2, v3)
                faces.append([v1, v2, v3])
                edges.add(tuple(sorted((v1, v2))))
                edges.add(tuple(sorted((v2, v3))))
                edges.add(tuple(sorted((v3, v1))))

                # 更新活动边界：v2成为内部点，将其移除
                active_front.popleft() # 移除 v1
                active_front.popleft() # 移除 v2
                active_front.appendleft(v1) # 把 v1 加回来

            elif op == 'E':  # End: 结束当前分量
                if len(active_front) == 3:
                    # 形成最后一个三角形
                    v1, v2, v3 = active_front[0], active_front[1], active_front[2]
                    faces.append([v1, v2, v3])
                    edges.add(tuple(sorted((v1, v2))))
                    edges.add(tuple(sorted((v2, v3))))
                    edges.add(tuple(sorted((v3, v1))))
                active_front.clear() # 清空边界
                break
        
        return self._build_graph_data(vertices, edges, faces, num_boundary_sides)
    
    def _build_graph_data(self, vertices: List[int], edges: Set[Tuple[int, int]], 
                         faces: List[List[int]], num_boundary_sides: int) -> Dict:
        """构建图数据结构"""
        num_nodes = len(vertices)
        
        # 确定最终的边界顶点
        # 初始边界顶点是0到N-1，但解码后实际的边界可能不同
        # 我们通过边的度数来重新计算
        node_degrees = defaultdict(int)
        for v1, v2 in edges:
            node_degrees[v1] += 1
            node_degrees[v2] += 1
        
        # 边界顶点是那些只属于一个边界环的顶点
        # 简化的方法是检查度数，但这不完全准确
        # 更准确的方法是追踪未被两个面共享的边
        edge_face_counts = defaultdict(int)
        for face in faces:
            for i in range(3):
                edge = tuple(sorted((face[i], face[(i+1)%3])))
                edge_face_counts[edge] += 1
        
        boundary_edges = {edge for edge, count in edge_face_counts.items() if count == 1}
        final_boundary_vertices = set()
        for v1, v2 in boundary_edges:
            final_boundary_vertices.add(v1)
            final_boundary_vertices.add(v2)

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
            vertices, edges, faces, list(final_boundary_vertices)
        )
        
        # 计算边特征
        edge_features = self._compute_edge_features(boundary_edges, edge_index)
        
        return {
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "node_features": node_features,
            "edge_features": edge_features,
            "faces": faces,
            "boundary_edges": boundary_edges
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
    
    def _compute_edge_features(self, boundary_edges: Set[Tuple[int, int]], 
                              edge_index: torch.Tensor) -> torch.Tensor:
        """计算边特征"""
        if edge_index.numel() == 0:
            return torch.empty((0, 1), dtype=torch.float)
        
        edge_features = []
        
        for i in range(edge_index.shape[1]):
            v1, v2 = edge_index[0, i].item(), edge_index[1, i].item()
            is_boundary_edge = tuple(sorted((v1, v2))) in boundary_edges
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
        
        boundary_edges = {tuple(sorted((i, (i + 1) % num_nodes))) for i in range(num_nodes)}

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
            "faces": [],
            "boundary_edges": boundary_edges
        }

class ProperPatternParser:
    """模式解析器"""
    def __init__(self, pattern_string: str, sides: int):
        self.pattern_string = pattern_string
        self.sides = sides
        self.decoder = EdgebreakerDecoder()
        
    def parse(self) -> Dict:
        """解析模式字符串生成图拓扑"""
        graph_data = self.decoder.decode_pattern_string(self.pattern_string, self.sides)
        
        if graph_data is None:
            graph_data = self.decoder._create_fallback_graph(self.sides)
        
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
            "is_boundary_edge": graph_data["edge_features"],
            "boundary_edges": graph_data["boundary_edges"] # 传递边界边信息
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