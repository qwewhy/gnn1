# File: src/use/visual_query_similar/mesh_loader.py
# 网格加载模块 / Mesh loading module

import trimesh
from pathlib import Path
from typing import Optional


class MeshLoader:
    """网格加载器 / Mesh Loader"""
    
    @staticmethod
    def load_mesh(mesh_path: str) -> Optional[trimesh.Trimesh]:
        """
        加载3D网格文件 / Load 3D mesh file
        
        Args:
            mesh_path: 网格文件路径 / Mesh file path
            
        Returns:
            加载的网格对象或None / Loaded mesh object or None
        """
        try:
            mesh = trimesh.load(mesh_path)
            
            # 如果是场景，取第一个几何体 / If it's a scene, get the first geometry
            if isinstance(mesh, trimesh.Scene):
                mesh = list(mesh.geometry.values())[0]
            
            print(f"✅ 成功加载网格: {Path(mesh_path).name}")
            print(f"   顶点数: {len(mesh.vertices)}")
            print(f"   面数: {len(mesh.faces)}")
            return mesh
            
        except Exception as e:
            print(f"❌ 网格加载失败: {e}")
            return None
    
    @staticmethod
    def find_mesh_files(model_dir: Path) -> list:
        """
        查找网格文件 / Find mesh files
        
        Args:
            model_dir: 模型目录路径 / Model directory path
            
        Returns:
            网格文件路径列表 / List of mesh file paths
        """
        mesh_files = []
        
        if model_dir.exists():
            mesh_files.extend(list(model_dir.glob('**/*.obj')))
            # 可以添加其他格式 / Can add other formats
            # mesh_files.extend(list(model_dir.glob('**/*.ply')))
            # mesh_files.extend(list(model_dir.glob('**/*.stl')))
        
        return mesh_files
    
    @staticmethod
    def validate_mesh(mesh: trimesh.Trimesh) -> bool:
        """
        验证网格是否有效 / Validate if mesh is valid
        
        Args:
            mesh: 网格对象 / Mesh object
            
        Returns:
            是否有效 / Whether it's valid
        """
        if mesh is None:
            return False
            
        if len(mesh.vertices) == 0:
            print("❌ 网格没有顶点")
            return False
            
        if len(mesh.faces) == 0:
            print("❌ 网格没有面")
            return False
            
        return True