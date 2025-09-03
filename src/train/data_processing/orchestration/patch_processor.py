# File: src/train/data_processing/orchestration/patch_processor.py
# 面片处理协调器 / Patch processor orchestrator

import trimesh
import numpy as np
from typing import List, Optional, Tuple, Dict
import logging

from ..mesh_processing.extraction.patch_extractor import ImprovedPatchExtractor
from ..mesh_processing.validation.patch_validator import PatchValidator
from ..encoding.topology.pattern_encoder import ProperPatternEncoder
from ..encoding.geometry.geometric_features import extract_geometric_features


class PatchProcessor:
    """面片处理协调器 - 统一管理面片提取、验证和编码"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patch_extractor = ImprovedPatchExtractor()
        self.patch_validator = PatchValidator()
        self.topology_encoder = ProperPatternEncoder()
    
    def extract_and_encode_patch(self, mesh: trimesh.Trimesh, 
                                face_adjacency_graph = None,
                                min_faces: int = 10, max_faces: int = 20) -> Optional[Tuple[str, str, int, Dict, Dict]]:
        """
        提取并编码面片，返回完整的拓扑和几何信息
        Extract and encode patch, returning complete topology and geometry information
        """
        try:
            # 1. 提取面片
            patch_faces = self.patch_extractor.extract_valid_patch(
                mesh, min_faces, max_faces
            )
            
            if patch_faces is None:
                return None
            
            # 2. 全面验证
            if not self.patch_validator.validate_patch_comprehensively(mesh, patch_faces):
                return None
            
            # 3. 编码面片
            return self._encode_patch_to_pattern(mesh, patch_faces)
            
        except Exception as e:
            self.logger.error(f"面片处理失败: {e}")
            return None
    
    def _encode_patch_to_pattern(self, mesh: trimesh.Trimesh, 
                                patch_face_indices: List[int]) -> Optional[Tuple[str, str, int, Dict, Dict]]:
        """
        编码几何面片，同时提取拓扑和几何特征
        Encode geometric patch while extracting topology and geometry features
        """
        try:
            # 1. 拓扑编码
            encoding_result = self.topology_encoder.encode_patch_to_pattern(mesh, patch_face_indices)
            
            if encoding_result is None:
                return None

            edgebreaker_encoding, num_sides = encoding_result
            canonical_form = edgebreaker_encoding.strip()

            # 2. 计算拓扑元数据
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

            # 3. 几何特征提取
            geometric_features = extract_geometric_features(mesh, patch_face_indices)

            if geometric_features is None:
                return None

            return edgebreaker_encoding, canonical_form, num_sides, topology_metadata, geometric_features

        except Exception as e:
            self.logger.error(f"编码面片失败: {e}")
            return None
