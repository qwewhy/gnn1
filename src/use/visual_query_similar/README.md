# 增强版3D网格面片可视化查询系统 / Enhanced 3D Mesh Patch Visual Query System

## 模块化重构说明 / Modular Refactoring Description

原始的 `visual_query.py` 文件（992行）已被重构为多个专门的模块，提高了代码的可维护性和可重用性。

The original `visual_query.py` file (992 lines) has been refactored into multiple specialized modules, improving code maintainability and reusability.

## 新的模块结构 / New Module Structure

```
src/use/visual_query_similar/
├── __init__.py              # 包初始化 / Package initialization
├── visual_query.py          # 向后兼容入口 / Backward compatible entry
├── main.py                  # 主入口模块 / Main entry module  
├── config.py                # 配置管理 / Configuration management
├── database_manager.py      # 数据库操作 / Database operations
├── mesh_loader.py           # 网格加载 / Mesh loading
├── visualizer_core.py       # 核心可视化 / Core visualization
├── html_generator.py        # HTML生成 / HTML generation
├── utils.py                 # 工具函数 / Utility functions
└── README.md               # 本文档 / This document
```

## 模块功能说明 / Module Functions

### 1. `config.py` - 配置管理 / Configuration Management
- 统一管理配置文件读取
- 路径验证和管理
- 项目结构导航

### 2. `database_manager.py` - 数据库管理 / Database Management
- 数据库连接和查询
- 面片信息获取
- 几何数据解析

### 3. `mesh_loader.py` - 网格加载 / Mesh Loading
- 3D网格文件加载
- 网格验证
- 文件查找工具

### 4. `visualizer_core.py` - 核心可视化 / Core Visualization
- 概览图创建
- 详细视图生成
- 3D场景渲染
- 面片可视化

### 5. `html_generator.py` - HTML生成 / HTML Generation
- 交互式仪表板生成
- CSS样式管理
- JavaScript功能

### 6. `utils.py` - 工具函数 / Utility Functions
- 环境设置
- 错误处理回退
- 用户交互工具
- 依赖验证

### 7. `main.py` - 主入口 / Main Entry
- 整合所有模块
- 主要的API接口
- 命令行界面

## 使用方法 / Usage

### 方式1：使用主函数 / Method 1: Use Main Function
```python
from src.use.visual_query_similar import main
main()
```

### 方式2：直接使用类 / Method 2: Use Class Directly
```python
from src.use.visual_query_similar import EnhancedMeshPatchVisualizer

# 创建可视化器
visualizer = EnhancedMeshPatchVisualizer("configs/config.yaml")

# 创建交互式仪表板
html_content, results = visualizer.create_interactive_dashboard("model/mesh.obj")

# 保存结果
output_path = visualizer.save_interactive_visualization(html_content, "result.html")
```

### 方式3：使用单独的模块 / Method 3: Use Individual Modules
```python
from src.use.visual_query_similar import (
    ConfigManager, 
    DatabaseManager, 
    MeshLoader, 
    VisualizerCore,
    HTMLGenerator
)

# 自定义使用各个模块
config = ConfigManager("configs/config.yaml")
db_manager = DatabaseManager(config.get_database_path())
# ... 等等
```

## 向后兼容性 / Backward Compatibility

原始的 `visual_query.py` 文件现在作为一个简单的包装器存在，保持与现有代码的兼容性。

The original `visual_query.py` file now exists as a simple wrapper, maintaining compatibility with existing code.

```python
# 原有的导入方式仍然有效 / Original import still works
from src.use.visual_query_similar.visual_query import EnhancedMeshPatchVisualizer
```

## 优势 / Advantages

1. **模块化** / **Modularity**: 每个模块专注于单一职责
2. **可维护性** / **Maintainability**: 更容易定位和修复问题
3. **可重用性** / **Reusability**: 模块可以独立使用
4. **可测试性** / **Testability**: 单独测试每个模块
5. **可扩展性** / **Extensibility**: 更容易添加新功能
6. **向后兼容** / **Backward Compatible**: 不破坏现有代码

## 代码行数对比 / Code Lines Comparison

| 文件 / File | 行数 / Lines | 功能 / Function |
|-------------|-------------|----------------|
| 原始 visual_query.py | 992 | 所有功能 / All functions |
| 新 visual_query.py | 62 | 入口包装器 / Entry wrapper |
| main.py | 234 | 主要逻辑 / Main logic |
| visualizer_core.py | 446 | 可视化核心 / Visualization core |
| html_generator.py | 251 | HTML生成 / HTML generation |
| database_manager.py | 152 | 数据库操作 / Database ops |
| utils.py | 193 | 工具函数 / Utilities |
| mesh_loader.py | 82 | 网格加载 / Mesh loading |
| config.py | 68 | 配置管理 / Config management |
| __init__.py | 41 | 包初始化 / Package init |

**总计 / Total**: 1529 行（重构后增加了文档和结构，但逻辑更清晰）
**Total**: 1529 lines (increased due to documentation and structure, but logic is clearer)