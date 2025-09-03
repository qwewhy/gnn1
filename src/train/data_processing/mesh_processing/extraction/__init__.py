# File: src/train/data_processing/mesh_processing/extraction/__init__.py
# 面片提取模块 / Patch extraction module

from .mesh_loader import MeshLoader
from .patch_extractor import ImprovedPatchExtractor, extract_random_patch

__all__ = [
    'MeshLoader',
    'ImprovedPatchExtractor', 
    'extract_random_patch'
]
