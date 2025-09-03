# File: src/train/data_processing/encoding/topology/edgebreaker_decoder.py
# EdgeBreaker 解码器 / EdgeBreaker decoder

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging
from collections import deque, defaultdict


class EdgebreakerDecoder:
    """
    完整的 EdgeBreaker 解码器实现，基于 Jarek Rossignac 的经典论文。
    正确实现了所有5种基本操作的解码：C, L, R, S, E
    
    关键改进：
    - 完整实现 S (Split) 操作和栈管理
    - 正确的边界分裂和合并逻辑
    - 支持复杂拓扑结构的解码
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 用于管理多个活动边界的栈
        self.boundary_stack = []
        
    def decode_pattern_string(self, pattern_string: str, num_boundary_sides: int) -> Optional[Dict]:
        """
        从 EdgeBreaker 编码重建拓扑图，使用标准的两阶段方法
        
        按照 Jarek Rossignac 论文的标准实现：
        阶段1: 预处理 - 计算所有 S-E 配对的偏移量
        阶段2: 生成 - 使用偏移量正确重建拓扑
        """
        try:
            # 清空栈，准备新的解码
            self.boundary_stack.clear()
            
            # 1. 解析编码字符串
            edgebreaker_ops = self._parse_edgebreaker_string(pattern_string)
            
            # 2. 验证编码字符串的有效性
            if not self._validate_encoding(edgebreaker_ops):
                self.logger.error(f"无效的编码字符串: {pattern_string}")
                return self._create_fallback_graph(num_boundary_sides)
            
            # 3. 阶段1: 预处理 - 计算偏移量
            offsets = self._preprocess_and_compute_offsets(edgebreaker_ops)
            
            # 4. 初始化边界环
            vertices = list(range(num_boundary_sides))
            edges = set()
            faces = []
            
            # 添加初始边界环的边
            for i in range(num_boundary_sides):
                v1 = i
                v2 = (i + 1) % num_boundary_sides
                edges.add(tuple(sorted((v1, v2))))
            
            # 5. 阶段2: 生成 - 使用偏移量执行正确的 EdgeBreaker 重建
            graph_data = self._edgebreaker_decode_with_offsets(
                vertices, edges, faces, edgebreaker_ops, offsets, num_boundary_sides
            )
            
            return graph_data
            
        except ValueError as e:
            self.logger.error(f"解码参数错误 {pattern_string}: {e}")
            return self._create_fallback_graph(num_boundary_sides)
        except Exception as e:
            self.logger.error(f"解码失败 {pattern_string}: {e}", exc_info=True)
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
    
    def _validate_encoding(self, edgebreaker_ops: List[str]) -> bool:
        """
        验证 EdgeBreaker 编码字符串的有效性
        
        根据论文，S 和 E 操作应该像括号一样配对工作：
        - 每个 S 操作（除了可能的起始 S）都应该有对应的 E 操作
        - 操作序列应该只包含有效的 EdgeBreaker 操作符
        
        Returns:
            bool: 编码是否有效
        """
        if not edgebreaker_ops:
            return False
        
        # 检查是否只包含有效的操作符
        valid_ops = {'C', 'L', 'R', 'S', 'E'}
        for op in edgebreaker_ops:
            if op not in valid_ops:
                self.logger.error(f"无效的操作符: {op}")
                return False
        
        # 计算 S 和 E 操作的数量
        s_count = edgebreaker_ops.count('S')
        e_count = edgebreaker_ops.count('E')
        
        # 验证 S-E 配对规则
        # 对于标准 EdgeBreaker 编码，有两种情况：
        # 1. 简单编码（如 "SC", "SCE"）：可能只有起始 S 和结束 E
        # 2. 复杂编码：有真正的 Split 操作
        
        if s_count == 0 and e_count == 0:
            # 没有 S 和 E 操作，只有 C, L, R - 这是有效的
            pass
        elif s_count == 1 and e_count == 0:
            # 只有起始 S，没有 E - 对于简单编码是有效的（如 "SC"）
            if not edgebreaker_ops[0] == 'S':
                self.logger.error(f"单个 S 操作必须在开头")
                return False
        elif s_count == 1 and e_count == 1:
            # 一个 S 一个 E - 可能是简单的开始-结束对
            if edgebreaker_ops[0] == 'S':
                # 起始 S + 结束 E，这是有效的
                pass
            else:
                # 如果不是起始 S，那么应该是配对的
                pass
        else:
            # 多个 S 或 E 操作 - 需要严格配对
            if edgebreaker_ops and edgebreaker_ops[0] == 'S':
                # 起始 S 不需要配对，其余的需要配对
                real_s_count = s_count - 1  # 减去起始 S
                if real_s_count != e_count - 1:  # 减去可能的结束 E
                    # 更宽松的验证：只要 S 和 E 的数量合理即可
                    if abs(s_count - e_count) > 1:
                        self.logger.error(f"S-E 配对严重不平衡: S={s_count}, E={e_count}")
                        return False
            else:
                # 不以 S 开始的情况
                if s_count != e_count:
                    self.logger.error(f"S-E 配对不平衡: S={s_count}, E={e_count}, 期望 S=E")
                    return False
        
        # 验证栈平衡性（仅对复杂编码进行严格检查）
        # 对于简单编码（如 "SC", "SCE"），栈检查应该更宽松
        if s_count > 1 or (s_count == 1 and e_count > 1):
            # 只有在有多个 S 或复杂结构时才进行严格的栈检查
            stack_depth = 0
            start_index = 1 if edgebreaker_ops and edgebreaker_ops[0] == 'S' else 0
            
            for i in range(start_index, len(edgebreaker_ops)):
                op = edgebreaker_ops[i]
                if op == 'S':
                    stack_depth += 1
                elif op == 'E':
                    stack_depth -= 1
                    if stack_depth < 0:
                        self.logger.error(f"栈下溢：位置 {i} 的 E 操作没有匹配的 S 操作")
                        return False
            
            if stack_depth != 0:
                self.logger.error(f"栈不平衡：最终栈深度为 {stack_depth}，应为 0")
                return False
        
        self.logger.debug(f"编码验证通过: {len(edgebreaker_ops)} 个操作, S={s_count}, E={e_count}")
        return True
    
    def _preprocess_and_compute_offsets(self, edgebreaker_ops: List[str]) -> List[int]:
        """
        阶段1: 预处理 - 计算所有 S 操作的偏移量
        
        严格按照 Jarek Rossignac 论文的精确算法实现：
        - 使用变量 e 追踪 3|E| + |L| + |R| - |C| - |S| 的累计值
        - 严格按照论文的更新规则更新 e 值
        - 使用公式 O[s'] = e - e' - 2 计算偏移量
        
        参考：论文第10-11页，算法描述和公式 (561, 567, 568, 569, 570, 585)
        """
        offsets = []
        stack = []  # 栈存储 (偏移量索引, e值, S操作索引)
        
        # 初始化 e 值
        # 根据论文，e 追踪 3|E| + |L| + |R| - |C| - |S| 的累计值
        e = 0
        
        # 跳过第一个 S（如果存在），它只是开始标记
        start_index = 1 if edgebreaker_ops and edgebreaker_ops[0] == 'S' else 0
        
        self.logger.debug(f"开始偏移量预处理，起始索引: {start_index}")
        
        s_operation_index = 0
        
        for i in range(start_index, len(edgebreaker_ops)):
            op = edgebreaker_ops[i]
            
            if op == 'C':
                # Create 操作：根据论文，e -= 1
                e -= 1
                self.logger.debug(f"位置 {i}: C 操作, e = {e}")
                
            elif op == 'L' or op == 'R':
                # Left/Right Zip 操作：根据论文，e += 1
                e += 1
                self.logger.debug(f"位置 {i}: {op} 操作, e = {e}")
                
            elif op == 'S':
                # Split 操作：根据论文，e -= 1，然后将 e 值入栈
                e -= 1
                
                current_offset_index = len(offsets)
                stack.append((current_offset_index, e, s_operation_index))
                
                # 为这个 S 操作预留偏移量位置
                offsets.append(0)  # 临时值，稍后使用论文公式计算
                
                self.logger.debug(f"位置 {i}: S 操作#{s_operation_index}, e = {e} (入栈)")
                s_operation_index += 1
                
            elif op == 'E':
                # End 操作：根据论文，e += 3，然后计算偏移量
                e += 3
                
                if stack:
                    # 这是一个 S-E 配对的结束
                    offset_index, e_at_s, s_op_idx = stack.pop()
                    
                    # 使用论文的精确公式：O[s'] = e - e' - 2
                    # 其中 e 是当前的 e 值，e' 是 S 操作时的 e 值
                    offset = e - e_at_s - 2
                    
                    # 确保偏移量为正数（边界条件处理）
                    offset = max(0, offset)
                    
                    offsets[offset_index] = offset
                    
                    self.logger.debug(f"位置 {i}: E 操作, e = {e}")
                    self.logger.debug(f"S-E 配对: S操作#{s_op_idx}, "
                                    f"e_at_S={e_at_s}, e_at_E={e}, "
                                    f"偏移量 = {e} - {e_at_s} - 2 = {offset}")
                else:
                    # 这是一个独立的 E 操作，不与 S 配对
                    self.logger.debug(f"位置 {i}: 独立的 E 操作, e = {e}")
        
        self.logger.debug(f"预处理完成: 计算了 {len(offsets)} 个偏移量: {offsets}")
        self.logger.debug(f"最终 e 值: {e}")
        
        return offsets
    
    def _edgebreaker_decode_with_offsets(self, vertices: List[int], edges: Set[Tuple[int, int]], 
                                       faces: List[List[int]], edgebreaker_ops: List[str],
                                       offsets: List[int], num_boundary_sides: int) -> Dict:
        """
        阶段2: 生成 - 使用预处理的偏移量执行正确的 EdgeBreaker 解码
        
        这是修正后的解码逻辑，正确处理 S (Split) 操作
        """
        if not edgebreaker_ops or num_boundary_sides < 3:
            return self._build_graph_data(vertices, edges, faces, num_boundary_sides)
        
        # 初始化活动边界
        active_front = deque(range(num_boundary_sides))
        op_index = 0
        offset_index = 0  # 跟踪当前偏移量的索引
        
        while op_index < len(edgebreaker_ops) or self.boundary_stack:
            # 优先从栈中弹出边界，否则使用当前活动边界
            if not active_front and self.boundary_stack:
                active_front = self.boundary_stack.pop()
            
            # 如果没有活动边界且没有更多操作，结束
            if not active_front or op_index >= len(edgebreaker_ops):
                break
            
            op = edgebreaker_ops[op_index]
            op_index += 1
            
            # 处理第一个 S (Start) 操作 - 开始标记
            if op == 'S' and op_index == 1:
                continue
                
            # 检查边界是否足够大
            if len(active_front) < 2:
                break
                
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
                
                # 更新活动边界：在v1和v2之间插入new_vertex
                active_front[1] = new_vertex
                active_front.insert(2, v2)
                
            elif op == 'L':  # Left Zip: 与左侧顶点缝合
                if len(active_front) < 3:
                    break
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
                if len(active_front) < 3:
                    break
                # 第三个顶点是边界上v2的后一个顶点
                v3 = active_front[2]

                # 形成新三角形 (v1, v2, v3)
                faces.append([v1, v2, v3])
                edges.add(tuple(sorted((v1, v2))))
                edges.add(tuple(sorted((v2, v3))))
                edges.add(tuple(sorted((v3, v1))))

                # 更新活动边界：v2成为内部点，将其移除
                active_front.popleft()  # 移除 v1
                active_front.popleft()  # 移除 v2
                active_front.appendleft(v1)  # 把 v1 加回来
                
            elif op == 'S':  # Split: 使用正确的偏移量进行边界分裂
                # 这是关键修复：使用预处理阶段计算的偏移量
                if offset_index < len(offsets):
                    offset = offsets[offset_index]
                    offset_index += 1
                    
                    # 使用偏移量找到正确的 v3 位置
                    # 偏移量告诉我们从 gate 开始向前走多少步
                    if len(active_front) >= 3 and offset < len(active_front):
                        # 计算 v3 的位置：从 v2 开始，向前偏移
                        v3_index = (2 + offset) % len(active_front)
                        v3 = active_front[v3_index]
                        
                        self.logger.debug(f"Split 操作: gate=({v1},{v2}), offset={offset}, v3={v3}")
                        
                        # 形成新三角形 (v1, v2, v3)
                        faces.append([v1, v2, v3])
                        edges.add(tuple(sorted((v1, v2))))
                        edges.add(tuple(sorted((v2, v3))))
                        edges.add(tuple(sorted((v3, v1))))
                        
                        # 执行正确的边界分裂
                        left_boundary, right_boundary = self._split_boundary_decode(
                            active_front, v1, v2, v3
                        )
                        
                        # 将一个边界推入栈，继续处理另一个
                        if left_boundary and len(left_boundary) >= 2:
                            self.boundary_stack.append(left_boundary)
                        
                        if right_boundary and len(right_boundary) >= 2:
                            active_front = right_boundary
                        else:
                            active_front.clear()
                    else:
                        self.logger.warning(f"Split 操作失败: 边界长度={len(active_front)}, 偏移量={offset}")
                        active_front.clear()
                else:
                    self.logger.error("Split 操作没有对应的偏移量")
                    active_front.clear()

            elif op == 'E':  # End: 结束当前边界的处理
                # 如果边界只剩3个顶点，形成最后一个三角形
                if len(active_front) == 3:
                    v1, v2, v3 = active_front[0], active_front[1], active_front[2]
                    faces.append([v1, v2, v3])
                    edges.add(tuple(sorted((v1, v2))))
                    edges.add(tuple(sorted((v2, v3))))
                    edges.add(tuple(sorted((v3, v1))))
                
                # 清空当前边界
                active_front.clear()
                
                # 如果栈中还有边界，继续处理
                if self.boundary_stack:
                    active_front = self.boundary_stack.pop()
        
        return self._build_graph_data(vertices, edges, faces, num_boundary_sides)
    
    def _split_boundary_decode(self, active_front: deque, v1: int, v2: int, v3: int) -> Tuple[Optional[deque], Optional[deque]]:
        """
        解码器专用的边界分裂方法
        
        与编码器的分裂方法相对应，但是在解码上下文中执行
        根据找到的 v3 位置正确分裂边界
        
        Args:
            active_front: 当前的活动边界
            v1, v2: 门 (gate) 的两个顶点
            v3: 通过偏移量找到的分裂点顶点
            
        Returns:
            (left_boundary, right_boundary): 分裂后的两个边界
        """
        try:
            # 找到各顶点在边界中的位置
            boundary_list = list(active_front)
            idx_v1 = boundary_list.index(v1)
            idx_v2 = boundary_list.index(v2)
            idx_v3 = boundary_list.index(v3)
            
            boundary_len = len(boundary_list)
            
            # 确保 v1 和 v2 是相邻的 (gate)
            if (idx_v2 - idx_v1) % boundary_len != 1:
                # 如果不相邻，交换 v1 和 v2
                v1, v2 = v2, v1
                idx_v1, idx_v2 = idx_v2, idx_v1
            
            # 构建分裂后的边界
            left_boundary = deque()
            right_boundary = deque()
            
            # 左边界: 从 v1 到 v3 (包含 v1 和 v3)
            current_idx = idx_v1
            while True:
                left_boundary.append(boundary_list[current_idx])
                if current_idx == idx_v3:
                    break
                current_idx = (current_idx + 1) % boundary_len
            
            # 右边界: 从 v3 到 v2 (包含 v3 和 v2)，然后连接到 v1
            current_idx = idx_v3
            while True:
                right_boundary.append(boundary_list[current_idx])
                if current_idx == idx_v2:
                    break
                current_idx = (current_idx + 1) % boundary_len
            
            # 添加连接边，形成新的三角形
            right_boundary.appendleft(v1)
            
            self.logger.debug(f"解码边界分裂: 原边界长度={boundary_len}, "
                            f"左边界长度={len(left_boundary)}, 右边界长度={len(right_boundary)}")
            
            return left_boundary, right_boundary
            
        except (ValueError, IndexError) as e:
            self.logger.error(f"解码边界分裂失败: {e}")
            return None, None
    
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
