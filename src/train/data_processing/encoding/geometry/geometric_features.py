# File: src/train/data_processing/encoding/geometry/geometric_features.py
# 几何特征提取器 / Geometric features extractor

import trimesh
import numpy as np
from typing import List, Dict, Optional
import logging


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
            
            # **CRITICAL FIX**: 将有序的全局顶点索引添加到特征中
            # 确保转换为Python int类型以支持JSON序列化
            features['ordered_boundary_vertex_indices'] = [int(v) for v in boundary_vertices]

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
        from ...mesh_processing.validation.geometry_validator import GeometryValidator
        
        geometry_validator = GeometryValidator()
        boundary_info = geometry_validator.compute_patch_boundary(mesh, patch_face_indices)
        
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
