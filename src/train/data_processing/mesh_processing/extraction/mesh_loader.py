# File: src/train/data_processing/mesh_processing/extraction/mesh_loader.py
# 网格加载器 / Mesh loader

import trimesh
import networkx as nx
import warnings
from pathlib import Path
from typing import Optional, Tuple
import logging


class MeshLoader:
    """网格文件加载和预处理"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_and_preprocess(self, obj_path: Path) -> Optional[Tuple[trimesh.Trimesh, nx.Graph]]:
        """
        加载并预处理.obj文件
        Load and preprocess .obj file
        """
        try:
            # 加载网格
            mesh = trimesh.load(obj_path, process=True)
            if not isinstance(mesh, trimesh.Trimesh):
                self.logger.warning(f"跳过非Trimesh对象: {obj_path.name}")
                return None

            # 预处理网格
            mesh.merge_vertices()
            
            # 使用新的API替代过时的方法
            try:
                mesh.update_faces(mesh.nondegenerate_faces())
                mesh.update_faces(mesh.unique_faces())
            except AttributeError:
                # 如果新API不可用，使用旧方法（带警告抑制）
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mesh.remove_degenerate_faces()
                    mesh.remove_duplicate_faces()

            # 构建面邻接图
            face_adjacency_graph = nx.from_edgelist(mesh.face_adjacency)
            
            return mesh, face_adjacency_graph
            
        except Exception as e:
            self.logger.error(f"加载失败 {obj_path.name}: {e}")
            return None
    
    def is_valid_mesh(self, mesh: trimesh.Trimesh) -> bool:
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
