# File: src/use/visual_query_similar/visual_query.py
# 可视化查询系统入口点 / Visual query system entry point
# 注意：此文件现在是对新模块化系统的简单包装器 / Note: This file is now a simple wrapper to the new modular system

"""
此文件保持与原有接口的兼容性，但现在使用新的模块化架构。
This file maintains compatibility with the original interface but now uses the new modular architecture.

新的模块化结构位于同一文件夹下：
New modular structure is located in the same folder:

- config.py: 配置管理 / Configuration management
- database_manager.py: 数据库操作 / Database operations  
- mesh_loader.py: 网格加载 / Mesh loading
- visualizer_core.py: 核心可视化 / Core visualization
- html_generator.py: HTML生成 / HTML generation
- utils.py: 工具函数 / Utility functions
- main.py: 主入口 / Main entry

推荐使用新的导入方式：
Recommended new import method:
    from src.use.visual_query_similar import EnhancedMeshPatchVisualizer, main
    或者 / or
    from src.use.visual_query_similar.main import main

使用示例 / Usage example:
    # 方式1：使用主函数 / Method 1: Use main function
    from src.use.visual_query_similar import main
    main()
    
    # 方式2：直接使用类 / Method 2: Use class directly
    from src.use.visual_query_similar import EnhancedMeshPatchVisualizer
    visualizer = EnhancedMeshPatchVisualizer("configs/config.yaml")
    result = visualizer.create_interactive_dashboard("path/to/mesh.obj")
"""

# 导入新的模块化系统 / Import new modular system
try:
    from .main import EnhancedMeshPatchVisualizer, main
except ImportError:
    from main import EnhancedMeshPatchVisualizer, main

# 为了向后兼容，重新导出主类 / Re-export main class for backward compatibility
__all__ = ['EnhancedMeshPatchVisualizer', 'main']


# 向后兼容的包装函数 / Backward compatible wrapper function
def create_visualizer(config_path: str) -> EnhancedMeshPatchVisualizer:
    """
    创建可视化器实例（向后兼容） / Create visualizer instance (backward compatible)
    
    Args:
        config_path: 配置文件路径 / Configuration file path
        
    Returns:
        可视化器实例 / Visualizer instance
    """
    return EnhancedMeshPatchVisualizer(config_path)


# 向后兼容的主函数入口 / Backward compatible main function entry
if __name__ == '__main__':
    print("🔄 使用新的模块化架构运行可视化系统...")
    print("🔄 Running visualization system with new modular architecture...")
    main()