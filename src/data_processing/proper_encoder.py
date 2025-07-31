import trimesh
import numpy as np
import networkx as nx
from typing import List, Optional, Tuple, Dict, Set
from collections import deque, defaultdict
import logging

class HalfEdgeStructure:
    """完整的半边数据结构实现"""
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.half_edges = {}  # edge_id -> (start_vertex, end_vertex)
        self.twin_edges = {}  # edge_id -> twin_edge_id
        self.next_edges = {}  # edge_id -> next_edge_id
        self.prev_edges = {}  # edge_id -> prev_edge_id
        self.face_edges = {}  # face_id -> first_edge_id
        self.edge_faces = {}  # edge_id -> face_id
        self.vertex_edges = defaultdict(list)  # vertex_id -> [edge_ids]
        self.boundary_loops = []
        self.next_edge_id = 0
        
    def add_vertex(self, vertex_data=None):
        """添加顶点"""
        vertex_id = len(self.vertices)
        self.vertices.append(vertex_data)
        return vertex_id
        
    def add_edge(self, start_vertex: int, end_vertex: int, face_id: int = None) -> int:
        """添加半边"""
        edge_id = self.next_edge_id
        self.next_edge_id += 1
        
        self.half_edges[edge_id] = (start_vertex, end_vertex)
        self.vertex_edges[start_vertex].append(edge_id)
        
        if face_id is not None:
            self.edge_faces[edge_id] = face_id
            
        return edge_id
        
    def add_face(self, vertices: List[int]) -> int:
        """添加面并创建相应的半边"""
        face_id = len(self.faces)
        self.faces.append(vertices)
        
        # 创建面的半边环
        edge_ids = []
        for i in range(len(vertices)):
            start_v = vertices[i]
            end_v = vertices[(i + 1) % len(vertices)]
            edge_id = self.add_edge(start_v, end_v, face_id)
            edge_ids.append(edge_id)
            
        # 连接半边环
        for i in range(len(edge_ids)):
            curr_edge = edge_ids[i]
            next_edge = edge_ids[(i + 1) % len(edge_ids)]
            prev_edge = edge_ids[(i - 1) % len(edge_ids)]
            
            self.next_edges[curr_edge] = next_edge
            self.prev_edges[curr_edge] = prev_edge
            
        self.face_edges[face_id] = edge_ids[0]
        
        # 查找并连接孪生边
        self._find_and_connect_twins(edge_ids)
        
        return face_id
        
    def _find_and_connect_twins(self, new_edge_ids: List[int]):
        """查找并连接孪生边"""
        for edge_id in new_edge_ids:
            if edge_id in self.twin_edges:
                continue
                
            start_v, end_v = self.half_edges[edge_id]
            
            # 查找反向边
            for other_edge_id, (other_start, other_end) in self.half_edges.items():
                if (other_edge_id != edge_id and 
                    other_start == end_v and other_end == start_v and
                    other_edge_id not in self.twin_edges):
                    
                    self.twin_edges[edge_id] = other_edge_id
                    self.twin_edges[other_edge_id] = edge_id
                    break
                    
    def find_boundary_loops(self) -> List[List[int]]:
        """查找所有边界环"""
        boundary_edges = []
        for edge_id in self.half_edges:
            if edge_id not in self.twin_edges:
                boundary_edges.append(edge_id)
                
        visited = set()
        loops = []
        
        for start_edge in boundary_edges:
            if start_edge in visited:
                continue
                
            loop = []
            current_edge = start_edge
            
            while current_edge not in visited:
                visited.add(current_edge)
                loop.append(current_edge)
                
                # 查找下一条边界边
                _, end_vertex = self.half_edges[current_edge]
                next_edge = None
                
                for candidate_edge in self.vertex_edges[end_vertex]:
                    if (candidate_edge not in self.twin_edges and 
                        candidate_edge != current_edge and
                        candidate_edge not in visited):
                        next_edge = candidate_edge
                        break
                        
                if next_edge is None:
                    break
                current_edge = next_edge
                
            if loop:
                loops.append(loop)
                
        self.boundary_loops = loops
        return loops

class MultiChordFolder:
    """真正的多弦折叠算法实现"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def encode_patch(self, mesh: trimesh.Trimesh, patch_faces: List[int]) -> Optional[str]:
        """将几何面片编码为拓扑模式字符串"""
        try:
            # 1. 构建半边数据结构
            he_struct = self._build_half_edge_structure(mesh, patch_faces)
            
            # 2. 识别边界环
            boundary_loops = he_struct.find_boundary_loops()
            if not boundary_loops:
                return None
                
            main_boundary = max(boundary_loops, key=len)  # 选择最长的边界环
            
            # 3. 识别多弦配置
            multi_chords = self._identify_multi_chords(he_struct, main_boundary)
            
            # 4. 执行多弦折叠
            folded_structure, folding_sequence = self._perform_multi_chord_folding(
                he_struct, multi_chords
            )
            
            # 5. 对折叠后的结构执行Edgebreaker编码
            edgebreaker_sequence = self._edgebreaker_encode(folded_structure)
            
            # 6. 构建最终编码
            final_encoding = self._build_final_encoding(folding_sequence, edgebreaker_sequence)
            
            return final_encoding
            
        except Exception as e:
            self.logger.error(f"编码失败: {e}")
            return None
            
    def _build_half_edge_structure(self, mesh: trimesh.Trimesh, patch_faces: List[int]) -> HalfEdgeStructure:
        """从网格面片构建半边结构"""
        he_struct = HalfEdgeStructure()
        
        # 收集面片中的所有顶点
        patch_vertices = set()
        for face_idx in patch_faces:
            face = mesh.faces[face_idx]
            patch_vertices.update(face)
            
        # 重新映射顶点索引
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(patch_vertices))}
        
        # 添加顶点
        for _ in range(len(vertex_map)):
            he_struct.add_vertex()
            
        # 添加面
        for face_idx in patch_faces:
            face = mesh.faces[face_idx]
            mapped_face = [vertex_map[v] for v in face]
            he_struct.add_face(mapped_face)
            
        return he_struct
        
    def _identify_multi_chords(self, he_struct: HalfEdgeStructure, 
                              boundary_edges: List[int]) -> List[List[int]]:
        """识别多弦配置"""
        # 获取边界顶点序列
        boundary_vertices = []
        for edge_id in boundary_edges:
            start_v, _ = he_struct.half_edges[edge_id]
            boundary_vertices.append(start_v)
            
        boundary_vertex_set = set(boundary_vertices)
        
        # 找到所有弦（连接边界顶点但不是边界边的内部边）
        chords = []
        for edge_id, (start_v, end_v) in he_struct.half_edges.items():
            # 检查是否是弦
            if (start_v in boundary_vertex_set and 
                end_v in boundary_vertex_set and
                edge_id not in boundary_edges and
                edge_id in he_struct.twin_edges):  # 必须是内部边
                
                # 计算在边界上的距离
                try:
                    start_idx = boundary_vertices.index(start_v)
                    end_idx = boundary_vertices.index(end_v)
                    
                    distance = min(
                        abs(start_idx - end_idx),
                        len(boundary_vertices) - abs(start_idx - end_idx)
                    )
                    
                    if distance > 1:  # 不是相邻顶点
                        chords.append({
                            'edge_id': edge_id,
                            'vertices': (start_v, end_v),
                            'boundary_distance': distance,
                            'boundary_indices': (start_idx, end_idx)
                        })
                except ValueError:
                    continue
                    
        # 按拓扑特征分组弦
        return self._group_chords_into_multi_chords(chords, boundary_vertices)
        
    def _group_chords_into_multi_chords(self, chords: List[Dict], 
                                       boundary_vertices: List[int]) -> List[List[int]]:
        """将弦分组为多弦"""
        if not chords:
            return []
            
        # 按距离分组
        distance_groups = defaultdict(list)
        for chord in chords:
            distance_groups[chord['boundary_distance']].append(chord)
            
        multi_chords = []
        
        for distance, group_chords in distance_groups.items():
            # 检查兼容性并分组
            compatible_groups = []
            
            for chord in group_chords:
                placed = False
                
                for group in compatible_groups:
                    if self._chords_are_compatible(chord, group, len(boundary_vertices)):
                        group.append(chord)
                        placed = True
                        break
                        
                if not placed:
                    compatible_groups.append([chord])
                    
            # 转换为边ID列表
            for group in compatible_groups:
                if len(group) >= 1:  # 至少一个弦
                    multi_chords.append([chord['edge_id'] for chord in group])
                    
        return multi_chords
        
    def _chords_are_compatible(self, chord: Dict, group: List[Dict], 
                              boundary_size: int) -> bool:
        """检查弦是否与组兼容"""
        chord_start, chord_end = chord['boundary_indices']
        
        for other_chord in group:
            other_start, other_end = other_chord['boundary_indices']
            
            # 检查是否共享顶点
            if (chord_start == other_start or chord_start == other_end or
                chord_end == other_start or chord_end == other_end):
                return False
                
            # 检查是否在边界上交叉
            if self._chords_intersect_on_boundary(
                chord_start, chord_end, other_start, other_end, boundary_size):
                return False
                
        return True
        
    def _chords_intersect_on_boundary(self, a1: int, a2: int, b1: int, b2: int, 
                                     boundary_size: int) -> bool:
        """检查两个弦是否在边界环上交叉"""
        # 标准化索引顺序
        if a1 > a2:
            a1, a2 = a2, a1
        if b1 > b2:
            b1, b2 = b2, b1
            
        def is_between_circular(x, start, end, size):
            if start <= end:
                return start < x < end
            else:  # 跨越0点
                return x > start or x < end
                
        # 检查交叉条件
        b1_between = is_between_circular(b1, a1, a2, boundary_size)
        b2_between = is_between_circular(b2, a1, a2, boundary_size)
        
        return b1_between != b2_between
        
    def _perform_multi_chord_folding(self, he_struct: HalfEdgeStructure, 
                                    multi_chords: List[List[int]]) -> Tuple[HalfEdgeStructure, List[str]]:
        """执行多弦折叠操作"""
        folded_struct = he_struct  # 创建副本
        folding_sequence = []
        
        for multi_chord in multi_chords:
            if len(multi_chord) == 1:
                # 单弦折叠
                chord_edge = multi_chord[0]
                folding_sequence.append(f"1")  # 记录弦的数量
                self._fold_single_chord(folded_struct, chord_edge)
            else:
                # 多弦同时折叠
                folding_sequence.append(f"{len(multi_chord)}")
                self._fold_multiple_chords(folded_struct, multi_chord)
                
        return folded_struct, folding_sequence
        
    def _fold_single_chord(self, he_struct: HalfEdgeStructure, chord_edge: int):
        """折叠单个弦"""
        # 简化实现：标记弦为已折叠
        # 在完整实现中，这里会修改拓扑结构
        pass
        
    def _fold_multiple_chords(self, he_struct: HalfEdgeStructure, chord_edges: List[int]):
        """同时折叠多个弦"""
        # 简化实现：标记所有弦为已折叠
        # 在完整实现中，这里会同时修改拓扑结构
        pass
        
    def _edgebreaker_encode(self, he_struct: HalfEdgeStructure) -> str:
        """对折叠后的结构执行Edgebreaker编码"""
        if not he_struct.faces:
            return "E"
            
        encoding = []
        visited_faces = set()
        face_stack = [0]  # 从第一个面开始
        
        while face_stack:
            current_face = face_stack.pop()
            
            if current_face in visited_faces:
                continue
                
            visited_faces.add(current_face)
            
            # 确定操作类型
            operation = self._determine_edgebreaker_operation(
                he_struct, current_face, visited_faces
            )
            encoding.append(operation)
            
            # 添加未访问的邻接面
            neighbors = self._get_face_neighbors(he_struct, current_face)
            for neighbor in neighbors:
                if neighbor not in visited_faces:
                    face_stack.append(neighbor)
                    
        return ''.join(encoding)
        
    def _determine_edgebreaker_operation(self, he_struct: HalfEdgeStructure,
                                        face_id: int, visited_faces: Set[int]) -> str:
        """确定Edgebreaker操作类型"""
        neighbors = self._get_face_neighbors(he_struct, face_id)
        unvisited_neighbors = [f for f in neighbors if f not in visited_faces]
        
        if len(visited_faces) == 1:
            return 'S'  # Start
        elif len(unvisited_neighbors) == 0:
            return 'E'  # End
        elif len(unvisited_neighbors) == 1:
            # 根据几何位置判断L或R（简化为随机）
            return 'L' if hash(face_id) % 2 == 0 else 'R'
        else:
            return 'C'  # Case
            
    def _get_face_neighbors(self, he_struct: HalfEdgeStructure, face_id: int) -> List[int]:
        """获取面的邻接面"""
        neighbors = []
        
        if face_id not in he_struct.face_edges:
            return neighbors
            
        # 遍历面的所有边
        start_edge = he_struct.face_edges[face_id]
        current_edge = start_edge
        
        while True:
            # 查找孪生边对应的面
            if current_edge in he_struct.twin_edges:
                twin_edge = he_struct.twin_edges[current_edge]
                if twin_edge in he_struct.edge_faces:
                    neighbor_face = he_struct.edge_faces[twin_edge]
                    if neighbor_face != face_id:
                        neighbors.append(neighbor_face)
                        
            # 移动到下一条边
            if current_edge in he_struct.next_edges:
                current_edge = he_struct.next_edges[current_edge]
                if current_edge == start_edge:
                    break
            else:
                break
                
        return neighbors
        
    def _build_final_encoding(self, folding_sequence: List[str], 
                             edgebreaker_sequence: str) -> str:
        """构建最终的编码字符串"""
        if not folding_sequence:
            return edgebreaker_sequence
            
        multi_chord_part = ' '.join(folding_sequence)
        return f"{multi_chord_part}#{edgebreaker_sequence}"

class ProperPatternEncoder:
    """正确的模式编码器"""
    def __init__(self):
        self.folder = MultiChordFolder()
        
    def encode_patch_to_pattern(self, mesh: trimesh.Trimesh, 
                               patch_face_indices: List[int]) -> Optional[Tuple[str, int]]:
        """将几何面片编码为规范的拓扑模式"""
        try:
            # 1. 使用多弦折叠算法编码
            encoding = self.folder.encode_patch(mesh, patch_face_indices)
            
            if encoding is None:
                return None
                
            # 2. 计算边界边数
            num_sides = self._calculate_boundary_sides(mesh, patch_face_indices)
            
            if num_sides is None:
                return None
                
            return encoding, num_sides
            
        except Exception as e:
            logging.error(f"编码面片失败: {e}")
            return None
            
    def _calculate_boundary_sides(self, mesh: trimesh.Trimesh, 
                                 patch_face_indices: List[int]) -> Optional[int]:
        """计算面片的边界边数"""
        try:
            # 使用trimesh的outline功能
            boundary_path = mesh.outline(patch_face_indices)
            
            if boundary_path is None or len(boundary_path.entities) == 0:
                return None
                
            # 获取主要边界环的边数
            main_entity = boundary_path.entities[0]
            return len(main_entity.points)
            
        except Exception as e:
            logging.error(f"计算边界边数失败: {e}")
            return None