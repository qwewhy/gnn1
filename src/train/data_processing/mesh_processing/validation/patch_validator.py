# File: src/train/data_processing/mesh_processing/validation/patch_validator.py
# 面片验证器 / Patch validator

import trimesh
import numpy as np
from typing import List, Dict, Optional
import logging
from .geometry_validator import GeometryValidator
from .topology_validator import TopologyValidator


class PatchValidator:
    """综合面片验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.geometry_validator = GeometryValidator()
        self.topology_validator = TopologyValidator()
    
    def validate_patch_comprehensively(self, mesh: trimesh.Trimesh, 
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
        if not self.geometry_validator.is_geometrically_connected(mesh, patch_faces):
            return False
        
        # 4. 检查边界有效性
        boundary_info = self.geometry_validator.compute_patch_boundary(mesh, patch_faces)
        if boundary_info is None:
            return False
        
        # 5. 检查拓扑有效性
        if not self.topology_validator.is_topologically_valid(mesh, patch_faces, boundary_info):
            return False
        
        return True
