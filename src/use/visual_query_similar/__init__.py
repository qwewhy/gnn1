# File: src/use/visual_query_similar/__init__.py
# 可视化查询模块初始化 / Visual query module initialization

"""
Enhanced Visual Query System for 3D Mesh Patches
增强版3D网格面片可视化查询系统

This package provides a modular system for visualizing and querying 3D mesh patches.
该包提供了一个模块化的系统用于可视化和查询3D网格面片。

Modules:
- config: Configuration management / 配置管理
- database_manager: Database operations / 数据库操作
- mesh_loader: Mesh loading utilities / 网格加载工具
- visualizer_core: Core visualization functions / 核心可视化功能
- html_generator: HTML generation for interactive dashboards / 交互式仪表板HTML生成
- utils: Utility functions / 工具函数
- main: Main entry point / 主入口点
"""

from .config import ConfigManager
from .database_manager import DatabaseManager
from .mesh_loader import MeshLoader
from .visualizer_core import VisualizerCore
from .html_generator import HTMLGenerator
from .utils import Utils
from .main import EnhancedMeshPatchVisualizer, main

__version__ = "1.0.0"
__author__ = "GNN Project Team"

__all__ = [
    'ConfigManager',
    'DatabaseManager', 
    'MeshLoader',
    'VisualizerCore',
    'HTMLGenerator',
    'Utils',
    'EnhancedMeshPatchVisualizer',
    'main'
]