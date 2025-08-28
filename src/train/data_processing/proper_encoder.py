# File: src/train/data_processing/proper_encoder.py
# 模式编码器 / Pattern encoder - 直接使用EdgeBreaker算法

import trimesh
import numpy as np
from typing import List, Optional, Tuple, Dict, Set
from collections import deque, defaultdict
import logging

class EdgeBreakerEncoder:
    """
    标准 EdgeBreaker 编码器的完整实现。
    通过维护一个 "主动边界" (active boundary) 和一个 "门" (gate) 来进行编码。
    
    标准 EdgeBreaker 算法的核心概念：
    - 主动边界 (Active Boundary): 已访问区域的边界，动态变化的顶点环路
    - 门 (Gate): 主动边界上的当前操作焦点边
    - C (Create): 发现新顶点，扩展边界
    - L (Left Zip): 向左缝合边界
    - R (Right Zip): 向右缝合边界  
    - E (End): 结束编码
    - S (Start/Split): 开始新分量或处理分裂
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def encode_patch(self, mesh: trimesh.Trimesh, patch_faces: List[int]) -> Optional[str]:
        """使用标准 EdgeBreaker 算法编码面片"""
        if not patch_faces or len(patch_faces) < 1:
            return "E"

        try:
            # 1. 为面片构建局部连接信息
            connectivity = self._build_patch_connectivity(mesh, patch_faces)
            if not connectivity:
                return "E"
            
            local_faces = connectivity['local_faces']
            edge_to_faces = connectivity['edge_to_faces']
            num_faces = len(local_faces)

            # 2. 初始化
            encoding = []
            visited_faces = np.zeros(num_faces, dtype=bool)
            
            # 3. 主循环，处理可能不连通的面片部分
            while np.any(~visited_faces):
                # 找到第一个未访问的面作为起点
                start_face_idx = np.where(~visited_faces)[0][0]
                
                # ---- 开始处理一个连通分量 ----
                # 标记起始面并初始化主动边界
                visited_faces[start_face_idx] = True
                start_face = local_faces[start_face_idx]
                
                # 'S' 操作代表开始一个新的网格/分量
                encoding.append('S')
                
                # 主动边界使用双端队列，方便左右操作
                active_boundary = deque(start_face)
                
                # 如果只有一个面，直接结束
                if num_faces == 1:
                    encoding.append('E')
                    break
                
                processed_edges_in_loop = 0
                while active_boundary:
                    # 如果在一个循环中处理的边数超过了边界长度，说明可能卡住了，强制跳出
                    if processed_edges_in_loop > len(active_boundary):
                         encoding.append('E') # 强制结束以避免死循环
                         break

                    # 将边界的第一个边作为 "门"
                    v1, v2 = active_boundary[0], active_boundary[1]
                    gate = tuple(sorted((v1, v2)))

                    # 查找与 "门" 共享的另一个面
                    shared_faces = edge_to_faces.get(gate, [])
                    opposite_face_idx = -1
                    for face_idx in shared_faces:
                        if not visited_faces[face_idx]:
                            opposite_face_idx = face_idx
                            break
                    
                    # Case A: 这是一个边界边，没有对面的未访问面
                    if opposite_face_idx == -1:
                        # 将 "门" 旋转到队列末尾，处理下一个边界边
                        active_boundary.rotate(-1)
                        processed_edges_in_loop += 1
                        continue

                    # Case B: 找到了一个未访问的面，进行编码
                    processed_edges_in_loop = 0 # 重置计数器
                    visited_faces[opposite_face_idx] = True
                    opposite_face = local_faces[opposite_face_idx]
                    
                    # 找到第三个顶点
                    v3 = next(v for v in opposite_face if v not in gate)

                    try:
                        # 检查 v3 是否在主动边界上
                        idx_v3 = active_boundary.index(v3)
                        
                        # ----- v3 在边界上: L, R, E 或 S 操作 -----
                        idx_v2 = active_boundary.index(v2)

                        # Right Zip (R)
                        if idx_v3 == (idx_v2 + 1) % len(active_boundary):
                            encoding.append('R')
                            active_boundary.remove(v2)
                        # Left Zip (L)
                        elif idx_v3 == (active_boundary.index(v1) - 1 + len(active_boundary)) % len(active_boundary):
                            encoding.append('L')
                            active_boundary.remove(v1)
                        # End (E) or Split (S) - 简化处理
                        else:
                            # 在这个简化版本中，我们假设面片是无洞的。
                            # 任何非邻接的连接都可能形成最后一个三角形或一个复杂的分裂。
                            # 我们在这里统一作为 End 处理。
                            encoding.append('E')
                            # 清空边界以结束当前分量的处理
                            active_boundary.clear()

                    except ValueError:
                        # ----- v3 不在边界上: C 操作 -----
                        encoding.append('C')
                        # 在 v1 和 v2 之间插入 v3
                        idx_v2 = active_boundary.index(v2)
                        active_boundary.insert(idx_v2, v3)

            return "".join(encoding)

        except Exception as e:
            self.logger.error(f"标准 EdgeBreaker 编码失败: {e}", exc_info=True)
            return None

    def _build_patch_connectivity(self, mesh: trimesh.Trimesh, patch_faces_indices: List[int]) -> Optional[Dict]:
        """为面片构建局部、从0开始的连接信息"""
        
        # 1. 找出所有涉及的顶点，并创建全局到局部的映射
        patch_global_verts = np.unique(mesh.faces[patch_faces_indices])
        global_to_local_v_map = {global_v: local_v for local_v, global_v in enumerate(patch_global_verts)}

        # 2. 创建本地化的面列表
        local_faces = []
        for face_idx in patch_faces_indices:
            local_face = [global_to_local_v_map[v] for v in mesh.faces[face_idx]]
            local_faces.append(local_face)

        # 3. 构建本地化的边 -> 面映射
        edge_to_faces = defaultdict(list)
        for i, face in enumerate(local_faces):
            for j in range(len(face)):
                v1 = face[j]
                v2 = face[(j + 1) % len(face)]
                edge = tuple(sorted((v1, v2)))
                edge_to_faces[edge].append(i)
        
        return {
            'local_faces': local_faces,
            'edge_to_faces': edge_to_faces
        }


class ProperPatternEncoder:
    """模式编码器 - 直接使用EdgeBreaker"""
    def __init__(self):
        self.encoder = EdgeBreakerEncoder()
        
    def encode_patch_to_pattern(self, mesh: trimesh.Trimesh, 
                               patch_face_indices: List[int]) -> Optional[Tuple[str, int]]:
        """将几何面片编码为EdgeBreaker模式"""
        try:
            # 1. 直接使用EdgeBreaker编码
            encoding = self.encoder.encode_patch(mesh, patch_face_indices)
            
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
        """计算面片的边界边数 - 改进版本，不依赖mesh.outline()"""
        try:
            # 使用和ImprovedPatchExtractor相同的方法
            edge_count = {}
            
            # 统计每条边被使用的次数
            for face_idx in patch_face_indices:
                face = mesh.faces[face_idx]
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                    edge_count[edge] = edge_count.get(edge, 0) + 1
            
            # 边界边只出现一次
            boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
            
            if not boundary_edges:
                return None
            
            # 构建边界图并追踪路径
            boundary_graph = {}
            for v1, v2 in boundary_edges:
                if v1 not in boundary_graph:
                    boundary_graph[v1] = []
                if v2 not in boundary_graph:
                    boundary_graph[v2] = []
                boundary_graph[v1].append(v2)
                boundary_graph[v2].append(v1)
            
            # 检查是否所有顶点都有度数2（形成简单环路）
            if not all(len(neighbors) == 2 for neighbors in boundary_graph.values()):
                return None
                
            # 边界顶点数等于边界边数
            return len(boundary_edges)
            
        except Exception as e:
            logging.error(f"计算边界边数失败: {e}")
            return None