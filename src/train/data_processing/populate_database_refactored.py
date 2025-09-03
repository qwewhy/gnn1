# File: src/train/data_processing/populate_database_refactored.py
# 重构后的数据库填充主入口 / Refactored database population main entry

"""
重构后的数据库填充脚本
Refactored database population script

这个脚本使用新的模块化架构，遵循单一职责原则：
- 数据库初始化
- 网格加载和预处理  
- 面片提取和验证
- 拓扑编码和几何特征提取
- 数据库填充协调

This script uses the new modular architecture following Single Responsibility Principle:
- Database initialization
- Mesh loading and preprocessing
- Patch extraction and validation
- Topology encoding and geometry feature extraction
- Database population orchestration
"""

import sys
from pathlib import Path

# 添加项目路径管理 - 与原始 populate_db.py 保持一致
try:
    from src.common.path_manager import setup_project_environment, get_database_path
    path_manager = setup_project_environment()
except ImportError:
    # 如果无法导入，使用fallback方法
    project_root_fallback = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root_fallback / 'src'))
    from src.common.path_manager import setup_project_environment, get_database_path
    path_manager = setup_project_environment()

# 现在可以安全地导入重构后的模块
from src.train.data_processing.orchestration.database_populator import DatabasePopulator


def main():
    """主函数"""
    print("🚀 启动重构后的数据库填充流程...")
    print("📋 使用模块化架构，遵循SOLID原则")
    
    populator = DatabasePopulator()
    populator.populate_database()
    
    print("✅ 数据库填充完成！")


if __name__ == '__main__':
    main()
