# File: src/train/data_processing/populate_db.py
# 数据库填充模块 / Database population module

import sys
import sqlite3
import trimesh
import numpy as np
import networkx as nx
from typing import List, Optional, Tuple, Dict, Set
import tqdm
import json
import importlib.util
import logging
from collections import deque, defaultdict
from pathlib import Path

# 添加项目路径管理
try:
    from src.common.path_manager import setup_project_environment, get_database_path
    path_manager = setup_project_environment()
except ImportError:
    # 如果无法导入，使用fallback方法
    project_root_fallback = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root_fallback / 'src'))
    from src.common.path_manager import setup_project_environment, get_database_path
    path_manager = setup_project_environment()


# 导入ProperPatternEncoder
def _load_encoder():
    """直接从模块文件加载ProperPatternEncoder"""
    encoder_path = Path(__file__).parent / "proper_encoder.py"
    spec = importlib.util.spec_from_file_location("proper_encoder", encoder_path)
    proper_encoder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proper_encoder_module)
    return proper_encoder_module.ProperPatternEncoder


ProperPatternEncoder = _load_encoder()


def setup_database(db_path: Path) -> sqlite3.Connection:
    """
    设置SQLite数据库，创建包含几何特征的patterns表
    """
    # 如果数据库存在，先删除以确保全新开始
    if db_path.exists():
        print(f"找到旧数据库 {db_path}，正在删除...")
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建增强的表结构，包含几何特征
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS patterns
                   (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       -- 拓扑信息
                       edgebreaker_encoding TEXT NOT NULL,
                       canonical_form TEXT NOT NULL,
                       sides INTEGER NOT NULL,
                       -- 基本元数据
                       complexity_score REAL DEFAULT 0.0,
                       num_vertices INTEGER DEFAULT 0,
                       num_faces INTEGER DEFAULT 0,
                       source_obj TEXT NOT NULL,
                       quality TEXT NOT NULL,
                       -- 几何特征（JSON格式存储）
                       boundary_vertices TEXT, -- 边界顶点坐标 [[x,y,z], ...]
                       vertex_normals TEXT, -- 顶点法线 [[nx,ny,nz], ...]
                       mean_curvatures TEXT, -- 平均曲率 [c1, c2, ...]
                       gaussian_curvatures TEXT, -- 高斯曲率 [g1, g2, ...]
                       edge_lengths TEXT, -- 边长度 [l1, l2, ...]
                       edge_curvatures TEXT, -- 边曲率 [ec1, ec2, ...]
                       -- 额外的统计信息
                       avg_curvature REAL, -- 平均曲率均值
                       curvature_variance REAL, -- 曲率方差
                       total_boundary_length REAL, -- 边界总长度
                       area REAL, -- 面片面积
                       UNIQUE(canonical_form, sides)
                   );
                   """)

    conn.commit()
    return conn


class ImprovedPatchExtractor:
    """改进的面片提取器，增加了多项验证"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_valid_patch(self, mesh: trimesh.Trimesh, 
                           min_faces: int = 10, max_faces: int = 20,
                           max_attempts: int = 50) -> Optional[List[int]]:
        """提取几何和拓扑上都有效的面片"""
        
        # 预处理网格
        if not self._is_mesh_valid(mesh):
            self.logger.warning("网格无效，跳过处理")
            return None
            
        face_adjacency_graph = self._build_robust_face_adjacency(mesh)
        
        for attempt in range(max_attempts):
            patch_faces = self._extract_single_patch(
                mesh, face_adjacency_graph, min_faces, max_faces
            )
            
            if patch_faces is None:
                continue
                
            # 多重验证
            if self._validate_patch_comprehensively(mesh, patch_faces):
                self.logger.info(f"成功提取有效patch，尝试次数: {attempt + 1}")
                return patch_faces
                
        self.logger.warning(f"经过{max_attempts}次尝试，未能提取到有效patch")
        return None
    
    def _is_mesh_valid(self, mesh: trimesh.Trimesh) -> bool:
        """验证网格基本有效性"""
        try:
            return (
                len(mesh.faces) >= 20 and  # 足够的面
                len(mesh.vertices) >= 10 and  # 足够的顶点
                not mesh.is_empty  # 非空
            )
        except:
            # 如果验证失败，进行基本检查
            return len(mesh.faces) >= 20 and len(mesh.vertices) >= 10
    
    def _build_robust_face_adjacency(self, mesh: trimesh.Trimesh) -> nx.Graph:
        """构建鲁棒的面邻接图"""
        try:
            # 使用trimesh内建的面邻接关系
            face_adjacency = mesh.face_adjacency
            graph = nx.Graph()
            graph.add_edges_from(face_adjacency)
            return graph
        except Exception as e:
            self.logger.error(f"构建面邻接图失败: {e}")
            # 备用方法：手动构建
            return self._manual_face_adjacency(mesh)
    
    def _manual_face_adjacency(self, mesh: trimesh.Trimesh) -> nx.Graph:
        """手动构建面邻接图（备用方法）"""
        graph = nx.Graph()
        faces = mesh.faces
        
        # 构建边到面的映射
        edge_to_faces = {}
        for face_idx, face in enumerate(faces):
            for i in range(len(face)):
                edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
        
        # 添加邻接关系
        for edge, face_list in edge_to_faces.items():
            if len(face_list) == 2:  # 共享边的两个面
                graph.add_edge(face_list[0], face_list[1])
        
        return graph
    
    def _extract_single_patch(self, mesh: trimesh.Trimesh, 
                             face_adjacency_graph: nx.Graph,
                             min_faces: int, max_faces: int) -> Optional[List[int]]:
        """提取单个面片（BFS）"""
        num_total_faces = len(mesh.faces)
        if num_total_faces < min_faces:
            return None
        
        # 选择连通性好的起始面
        start_face_idx = self._select_good_start_face(
            mesh, face_adjacency_graph, num_total_faces
        )
        
        # BFS扩展
        queue = deque([start_face_idx])
        visited = {start_face_idx}
        patch_faces = [start_face_idx]
        
        while queue and len(patch_faces) < max_faces:
            current_face = queue.popleft()
            
            neighbors = list(face_adjacency_graph.neighbors(current_face))
            # 按某种策略排序邻居（比如面积、法向量相似度）
            neighbors = self._sort_neighbors_by_quality(mesh, current_face, neighbors)
            
            for neighbor in neighbors:
                if neighbor not in visited and len(patch_faces) < max_faces:
                    visited.add(neighbor)
                    patch_faces.append(neighbor)
                    queue.append(neighbor)
        
        return patch_faces if len(patch_faces) >= min_faces else None
    
    def _select_good_start_face(self, mesh: trimesh.Trimesh, 
                               graph: nx.Graph, num_faces: int) -> int:
        """选择连通性好的起始面"""
        # 倾向选择度数适中的面作为起始点
        degrees = dict(graph.degree())
        
        # 过滤掉度数过低或过高的面
        good_faces = [
            face_idx for face_idx, degree in degrees.items() 
            if 2 <= degree <= 6
        ]
        
        if good_faces:
            return np.random.choice(good_faces)
        else:
            return np.random.randint(0, num_faces)
    
    def _sort_neighbors_by_quality(self, mesh: trimesh.Trimesh, 
                                  current_face: int, neighbors: List[int]) -> List[int]:
        """按质量排序邻居面"""
        if not neighbors:
            return neighbors
            
        try:
            current_normal = mesh.face_normals[current_face]
            
            # 计算法向量相似度
            scores = []
            for neighbor in neighbors:
                neighbor_normal = mesh.face_normals[neighbor]
                similarity = np.dot(current_normal, neighbor_normal)
                scores.append((neighbor, similarity))
            
            # 按相似度降序排序
            scores.sort(key=lambda x: x[1], reverse=True)
            return [face_idx for face_idx, _ in scores]
            
        except:
            return neighbors  # 如果计算失败，返回原顺序
    
    def _validate_patch_comprehensively(self, mesh: trimesh.Trimesh, 
                                       patch_faces: List[int]) -> bool:
        """综合验证patch的有效性"""
        
        # 1. 基本检查
        if not patch_faces or len(patch_faces) < 3:
            return False
        
        # 2. 检查面索引有效性
        max_face_idx = len(mesh.faces) - 1
        if any(face_idx < 0 or face_idx > max_face_idx for face_idx in patch_faces):
            return False
        
        # 3. 检查几何连通性
        if not self._is_geometrically_connected(mesh, patch_faces):
            return False
        
        # 4. 检查边界有效性
        boundary_info = self._compute_patch_boundary(mesh, patch_faces)
        if boundary_info is None:
            return False
        
        # 5. 检查拓扑有效性
        if not self._is_topologically_valid(mesh, patch_faces, boundary_info):
            return False
        
        return True
    
    def _is_geometrically_connected(self, mesh: trimesh.Trimesh, 
                                   patch_faces: List[int]) -> bool:
        """检查面片是否几何连通"""
        try:
            # 提取patch的顶点
            patch_vertices = set()
            for face_idx in patch_faces:
                face = mesh.faces[face_idx]
                patch_vertices.update(face)
            
            # 构建patch内部的顶点连通图
            vertex_graph = nx.Graph()
            for face_idx in patch_faces:
                face = mesh.faces[face_idx]
                for i in range(len(face)):
                    v1, v2 = face[i], face[(i + 1) % len(face)]
                    vertex_graph.add_edge(v1, v2)
            
            # 检查连通性
            return nx.is_connected(vertex_graph)
            
        except Exception as e:
            self.logger.warning(f"几何连通性检查失败: {e}")
            return True  # 如果检查失败，假设连通
    
    def _compute_patch_boundary(self, mesh: trimesh.Trimesh, 
                               patch_faces: List[int]) -> Optional[Dict]:
        """计算面片边界信息"""
        try:
            patch_face_set = set(patch_faces)
            boundary_edges = []
            
            # 找出边界边（只属于一个面的边）
            edge_count = {}
            for face_idx in patch_faces:
                face = mesh.faces[face_idx]
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                    edge_count[edge] = edge_count.get(edge, 0) + 1
            
            # 边界边只出现一次
            boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
            
            if not boundary_edges:
                return None
            
            # 构建边界路径
            boundary_vertices = self._trace_boundary_path(boundary_edges)
            
            return {
                'edges': boundary_edges,
                'vertices': boundary_vertices,
                'num_boundary_vertices': len(boundary_vertices) if boundary_vertices else 0
            }
            
        except Exception as e:
            self.logger.error(f"边界计算失败: {e}")
            return None
    
    def _trace_boundary_path(self, boundary_edges: List[Tuple[int, int]]) -> Optional[List[int]]:
        """追踪边界路径，形成有序的边界顶点序列"""
        if not boundary_edges:
            return None
            
        # 构建边界图
        boundary_graph = nx.Graph()
        boundary_graph.add_edges_from(boundary_edges)
        
        # 检查是否形成简单环路
        if not all(degree == 2 for _, degree in boundary_graph.degree()):
            self.logger.warning("边界不形成简单环路")
            return None
        
        # 追踪路径
        try:
            start_vertex = boundary_edges[0][0]
            path = [start_vertex]
            current = start_vertex
            prev = None
            
            while True:
                neighbors = [n for n in boundary_graph.neighbors(current) if n != prev]
                if not neighbors:
                    break
                    
                next_vertex = neighbors[0]
                if next_vertex == start_vertex:  # 回到起点
                    break
                    
                path.append(next_vertex)
                prev = current
                current = next_vertex
                
                # 防止无限循环
                if len(path) > len(boundary_edges) + 1:
                    break
            
            return path if len(path) >= 3 else None
            
        except Exception as e:
            self.logger.error(f"边界路径追踪失败: {e}")
            return None
    
    def _is_topologically_valid(self, mesh: trimesh.Trimesh, 
                               patch_faces: List[int], boundary_info: Dict) -> bool:
        """检查拓扑有效性"""
        try:
            num_boundary_vertices = boundary_info['num_boundary_vertices']
            
            # 边界应该至少有3个顶点
            if num_boundary_vertices < 3:
                return False
            
            # 边界顶点数不应该太大（相对于面片大小）
            if num_boundary_vertices > len(patch_faces) * 2:
                return False
            
            # 使用欧拉公式检查：V - E + F = 2（对于球面拓扑）
            # 这里是简化检查
            patch_vertices = set()
            patch_edges = set()
            
            for face_idx in patch_faces:
                face = mesh.faces[face_idx]
                patch_vertices.update(face)
                for i in range(len(face)):
                    edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
                    patch_edges.add(edge)
            
            V = len(patch_vertices)
            E = len(patch_edges)
            F = len(patch_faces)
            
            euler_char = V - E + F
            # 对于有边界的面片，欧拉特征数应该是1
            if not (0 <= euler_char <= 2):
                self.logger.warning(f"拓扑检查失败: V={V}, E={E}, F={F}, χ={euler_char}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"拓扑有效性检查失败: {e}")
            return True  # 如果检查失败，假设有效


def extract_random_patch(mesh: trimesh.Trimesh, face_adjacency_graph: nx.Graph,
                         min_faces: int = 10, max_faces: int = 20) -> Optional[List[int]]:
    """使用改进的面片提取器"""
    extractor = ImprovedPatchExtractor()
    return extractor.extract_valid_patch(mesh, min_faces, max_faces)


class ImprovedGeometricFeatureExtractor:
    """改进的几何特征提取器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_features_robust(self, mesh: trimesh.Trimesh, 
                               patch_faces: List[int], 
                               boundary_info: Dict) -> Optional[Dict]:
        """鲁棒的几何特征提取"""
        try:
            boundary_vertices = boundary_info['vertices']
            if not boundary_vertices:
                return None
            
            features = {}
            
            # 1. 提取边界顶点坐标
            features['boundary_vertices'] = [
                mesh.vertices[v].tolist() for v in boundary_vertices
            ]
            
            # 2. 计算边界法线（通过相邻面的法线插值）
            features['vertex_normals'] = self._compute_boundary_normals(
                mesh, patch_faces, boundary_vertices
            )
            
            # 3. 计算曲率（使用鲁棒方法）
            curvatures = self._compute_robust_curvatures(
                mesh, boundary_vertices
            )
            features.update(curvatures)
            
            # 4. 计算边几何特征
            edge_features = self._compute_edge_features(
                mesh, boundary_vertices
            )
            features.update(edge_features)
            
            # 5. 计算全局统计特征
            global_features = self._compute_global_features(
                mesh, patch_faces, features
            )
            features.update(global_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"几何特征提取失败: {e}")
            return None
    
    def _compute_boundary_normals(self, mesh: trimesh.Trimesh, 
                                 patch_faces: List[int],
                                 boundary_vertices: List[int]) -> List[List[float]]:
        """计算边界顶点法线"""
        try:
            vertex_normals = []
            
            for vertex_idx in boundary_vertices:
                # 找到包含此顶点的面片中的面
                adjacent_faces = []
                for face_idx in patch_faces:
                    face = mesh.faces[face_idx]
                    if vertex_idx in face:
                        adjacent_faces.append(face_idx)
                
                if adjacent_faces:
                    # 计算相邻面法线的平均值
                    normal_sum = np.zeros(3)
                    for face_idx in adjacent_faces:
                        normal_sum += mesh.face_normals[face_idx]
                    
                    normal = normal_sum / len(adjacent_faces)
                    # 归一化
                    norm = np.linalg.norm(normal)
                    if norm > 1e-10:
                        normal = normal / norm
                else:
                    normal = np.array([0.0, 0.0, 1.0])  # 默认法线
                
                vertex_normals.append(normal.tolist())
            
            return vertex_normals
            
        except Exception as e:
            self.logger.error(f"法线计算失败: {e}")
            return [[0.0, 0.0, 1.0]] * len(boundary_vertices)
    
    def _compute_robust_curvatures(self, mesh: trimesh.Trimesh, 
                                  boundary_vertices: List[int]) -> Dict:
        """计算鲁棒的曲率"""
        try:
            mean_curvatures = []
            gaussian_curvatures = []
            
            # 自适应半径
            mesh_scale = mesh.bounding_box.extents.max()
            radius = max(mesh_scale / 100.0, mesh.scale / 50.0)
            
            for vertex_idx in boundary_vertices:
                try:
                    vertex_pos = mesh.vertices[vertex_idx:vertex_idx+1]
                    
                    # 使用trimesh的离散曲率计算
                    mean_curv = trimesh.curvature.discrete_mean_curvature_measure(
                        mesh, vertex_pos, radius
                    )[0]
                    
                    gaussian_curv = trimesh.curvature.discrete_gaussian_curvature_measure(
                        mesh, vertex_pos, radius
                    )[0]
                    
                    # 数值稳定性处理
                    mean_curv = float(np.clip(mean_curv, -1e6, 1e6))
                    gaussian_curv = float(np.clip(gaussian_curv, -1e6, 1e6))
                    
                    # 检查NaN/Inf
                    if np.isnan(mean_curv) or np.isinf(mean_curv):
                        mean_curv = 0.0
                    if np.isnan(gaussian_curv) or np.isinf(gaussian_curv):
                        gaussian_curv = 0.0
                    
                    mean_curvatures.append(mean_curv)
                    gaussian_curvatures.append(gaussian_curv)
                    
                except Exception:
                    # 如果单个顶点计算失败，使用默认值
                    mean_curvatures.append(0.0)
                    gaussian_curvatures.append(0.0)
            
            return {
                'mean_curvatures': mean_curvatures,
                'gaussian_curvatures': gaussian_curvatures
            }
            
        except Exception as e:
            self.logger.error(f"曲率计算失败: {e}")
            num_vertices = len(boundary_vertices)
            return {
                'mean_curvatures': [0.0] * num_vertices,
                'gaussian_curvatures': [0.0] * num_vertices
            }
    
    def _compute_edge_features(self, mesh: trimesh.Trimesh, 
                              boundary_vertices: List[int]) -> Dict:
        """计算边特征"""
        try:
            edge_lengths = []
            edge_curvatures = []
            
            num_vertices = len(boundary_vertices)
            
            for i in range(num_vertices):
                v1_idx = boundary_vertices[i]
                v2_idx = boundary_vertices[(i + 1) % num_vertices]
                
                # 边长
                edge_length = np.linalg.norm(
                    mesh.vertices[v2_idx] - mesh.vertices[v1_idx]
                )
                edge_length = max(edge_length, 1e-10)  # 防止零长度
                edge_lengths.append(float(edge_length))
                
                # 边曲率（简化计算）
                edge_curv = 0.0  # 可以根据需要改进计算方法
                edge_curvatures.append(edge_curv)
            
            return {
                'edge_lengths': edge_lengths,
                'edge_curvatures': edge_curvatures
            }
            
        except Exception as e:
            self.logger.error(f"边特征计算失败: {e}")
            num_vertices = len(boundary_vertices)
            return {
                'edge_lengths': [1.0] * num_vertices,
                'edge_curvatures': [0.0] * num_vertices
            }
    
    def _compute_global_features(self, mesh: trimesh.Trimesh, 
                                patch_faces: List[int], features: Dict) -> Dict:
        """计算全局统计特征"""
        try:
            # 统计特征
            mean_curvatures = features.get('mean_curvatures', [])
            edge_lengths = features.get('edge_lengths', [])
            
            # 安全计算统计量
            if mean_curvatures:
                mean_curvatures_array = np.array(mean_curvatures)
                # 过滤异常值
                valid_curvatures = mean_curvatures_array[np.isfinite(mean_curvatures_array)]
                
                avg_curvature = float(np.mean(valid_curvatures)) if len(valid_curvatures) > 0 else 0.0
                curvature_variance = float(np.var(valid_curvatures)) if len(valid_curvatures) > 0 else 0.0
            else:
                avg_curvature = 0.0
                curvature_variance = 0.0
            
            total_boundary_length = float(np.sum(edge_lengths)) if edge_lengths else 0.0
            
            # 计算面片面积
            area = 0.0
            for face_idx in patch_faces:
                try:
                    face = mesh.faces[face_idx]
                    if len(face) >= 3:
                        v0, v1, v2 = mesh.vertices[face[:3]]
                        triangle_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                        if np.isfinite(triangle_area) and triangle_area > 0:
                            area += triangle_area
                except:
                    continue
            
            return {
                'avg_curvature': avg_curvature,
                'curvature_variance': curvature_variance,
                'total_boundary_length': total_boundary_length,
                'area': float(area)
            }
            
        except Exception as e:
            self.logger.error(f"全局特征计算失败: {e}")
            return {
                'avg_curvature': 0.0,
                'curvature_variance': 0.0,
                'total_boundary_length': 0.0,
                'area': 0.0
            }


def extract_geometric_features(mesh: trimesh.Trimesh, patch_face_indices: List[int]) -> Optional[Dict]:
    """
    从网格面片中提取几何特征 - 改进版本
    """
    try:
        # 首先尝试计算边界信息
        extractor = ImprovedPatchExtractor()
        boundary_info = extractor._compute_patch_boundary(mesh, patch_face_indices)
        
        if boundary_info is None:
            print("无法计算边界信息，跳过该面片")
            return None
        
        # 使用改进的几何特征提取器
        feature_extractor = ImprovedGeometricFeatureExtractor()
        geometric_features = feature_extractor.extract_features_robust(
            mesh, patch_face_indices, boundary_info
        )
        
        return geometric_features

    except Exception as e:
        print(f"提取几何特征失败: {e}")
        return None


def encode_patch_to_pattern(mesh: trimesh.Trimesh, patch_face_indices: List[int]) -> Optional[
    Tuple[str, str, int, Dict, Dict]]:
    """
    编码几何面片，同时提取拓扑和几何特征
    """
    try:
        encoder = ProperPatternEncoder()
        encoding_result = encoder.encode_patch_to_pattern(mesh, patch_face_indices)

        if encoding_result is None:
            return None

        edgebreaker_encoding, num_sides = encoding_result
        canonical_form = edgebreaker_encoding.strip()

        patch_faces = mesh.faces[patch_face_indices]
        unique_vertices = np.unique(patch_faces)
        num_faces = len(patch_faces)
        num_vertices = len(unique_vertices)
        complexity_score = num_faces / max(num_vertices, 1)

        topology_metadata = {
            'complexity_score': complexity_score,
            'num_vertices': num_vertices,
            'num_faces': num_faces
        }

        geometric_features = extract_geometric_features(mesh, patch_face_indices)

        if geometric_features is None:
            return None

        return edgebreaker_encoding, canonical_form, num_sides, topology_metadata, geometric_features

    except Exception as e:
        print(f"编码面片失败: {e}")
        return None


def main():
    """
    处理所有.obj文件并填充数据库
    """
    # 使用统一的路径管理
    db_path = get_database_path()
    model_dir = path_manager.model_dir
    patches_per_model = 100

    print(f"🗄️ 设置数据库: {db_path}")
    print(f"📁 模型目录: {model_dir}")
    
    conn = setup_database(db_path)
    cursor = conn.cursor()

    obj_files = list(model_dir.glob("**/*.obj"))
    print(f"找到 {len(obj_files)} 个模型文件")

    for obj_path in tqdm.tqdm(obj_files, desc="处理模型"):
        quality = obj_path.parent.name
        if quality not in ['new', 'old']:
            tqdm.tqdm.write(f"跳过非预期目录中的文件: {obj_path}")
            continue

        try:
            mesh = trimesh.load(obj_path, process=True)
            if not isinstance(mesh, trimesh.Trimesh):
                tqdm.tqdm.write(f"跳过非Trimesh对象: {obj_path.name}")
                continue

            mesh.merge_vertices()
            # 使用新的API替代过时的方法
            try:
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.update_faces(mesh.unique_faces())
            except AttributeError:
                # 如果新API不可用，使用旧方法（带警告抑制）
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mesh.remove_degenerate_faces()
                    mesh.remove_duplicate_faces()

            face_adjacency_graph = nx.from_edgelist(mesh.face_adjacency)

        except Exception as e:
            tqdm.tqdm.write(f"加载失败 {obj_path.name}: {e}")
            continue

        successful_patches = 0
        for _ in range(patches_per_model):
            patch_indices = extract_random_patch(mesh, face_adjacency_graph)
            if not patch_indices:
                continue

            encoding_result = encode_patch_to_pattern(mesh, patch_indices)
            if not encoding_result:
                continue

            edgebreaker_encoding, canonical_form, sides, topology_metadata, geometric_features = encoding_result

            try:
                cursor.execute("""
                               INSERT INTO patterns (edgebreaker_encoding, canonical_form, sides,
                                                     complexity_score, num_vertices, num_faces,
                                                     source_obj, quality,
                                                     boundary_vertices, vertex_normals,
                                                     mean_curvatures, gaussian_curvatures,
                                                     edge_lengths, edge_curvatures,
                                                     avg_curvature, curvature_variance,
                                                     total_boundary_length, area)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                               """, (
                                   edgebreaker_encoding, canonical_form, sides,
                                   topology_metadata['complexity_score'],
                                   topology_metadata['num_vertices'],
                                   topology_metadata['num_faces'],
                                   obj_path.name, quality,
                                   json.dumps(geometric_features['boundary_vertices']),
                                   json.dumps(geometric_features['vertex_normals']),
                                   json.dumps(geometric_features['mean_curvatures']),
                                   json.dumps(geometric_features['gaussian_curvatures']),
                                   json.dumps(geometric_features['edge_lengths']),
                                   json.dumps(geometric_features['edge_curvatures']),
                                   geometric_features['avg_curvature'],
                                   geometric_features['curvature_variance'],
                                   geometric_features['total_boundary_length'],
                                   geometric_features['area']
                               ))

                successful_patches += 1

            except sqlite3.IntegrityError:
                pass
            except Exception as e:
                tqdm.tqdm.write(f"插入失败 for {obj_path.name}: {e}")

        if successful_patches > 0:
            tqdm.tqdm.write(f"成功从 {obj_path.name} 提取 {successful_patches} 个面片")

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM patterns")
    total_patterns = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM patterns WHERE quality='new'")
    new_patterns = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM patterns WHERE quality='old'")
    old_patterns = cursor.fetchone()[0]

    print(f"\n数据库填充完成！")
    print(f"总模式数: {total_patterns}")
    print(f"'new'模式数: {new_patterns}")
    print(f"'old'模式数: {old_patterns}")

    conn.close()


if __name__ == '__main__':
    main()
