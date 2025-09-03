"""
数据处理模块 - 重构版本（遵循单一职责原则）
Data processing module - Refactored version (following Single Responsibility Principle)
"""

# 导入重构后的模块
from .database import setup_database
from .mesh_processing import MeshLoader, ImprovedPatchExtractor, extract_random_patch, PatchValidator
from .encoding import (
    EdgeBreakerEncoder, EdgebreakerDecoder,
    ProperPatternEncoder, ProperPatternParser,
    ImprovedGeometricFeatureExtractor, extract_geometric_features
)
from .orchestration import PatchProcessor, DatabasePopulator

print("使用重构后的模块化实现")

# 有条件导入PyG相关模块
try:
    from .pyg_dataset import PatchDataset
    from .triplet_generator import TripletGenerator

    _HAS_TORCH_GEOMETRIC = True
except ImportError as e:
    print(f"PyTorch Geometric相关模块导入失败: {e}")
    _HAS_TORCH_GEOMETRIC = False
    PatchDataset = None
    TripletGenerator = None

__all__ = [
    # 数据库模块
    'setup_database',
    
    # 网格处理模块
    'MeshLoader',
    'ImprovedPatchExtractor',
    'extract_random_patch',
    'PatchValidator',
    
    # 编码模块
    'EdgeBreakerEncoder',
    'EdgebreakerDecoder', 
    'ProperPatternEncoder',
    'ProperPatternParser',
    'ImprovedGeometricFeatureExtractor',
    'extract_geometric_features',
    
    # 协调模块
    'PatchProcessor',
    'DatabasePopulator',
]

# 有条件添加PyG模块
if _HAS_TORCH_GEOMETRIC:
    __all__.extend([
        'PatchDataset',
        'TripletGenerator',
    ])

# 版本信息
__version__ = "2.0.0"  # 重构版本号
__status__ = "Refactored with SOLID Principles" if _HAS_TORCH_GEOMETRIC else "Refactored Limited"


def verify_dependencies():
    """验证依赖完整性"""
    missing_deps = []

    try:
        import torch
    except ImportError:
        missing_deps.append("torch")

    try:
        import torch_geometric
    except ImportError:
        missing_deps.append("torch_geometric")

    try:
        import trimesh
    except ImportError:
        missing_deps.append("trimesh")

    try:
        import networkx
    except ImportError:
        missing_deps.append("networkx")

    if missing_deps:
        print(f"缺少依赖: {', '.join(missing_deps)}")
        return False

    print("所有依赖验证通过")
    return True


def get_implementation_status():
    """获取实现状态"""
    status = {
        "encoder": "基础实现",  # 使用基础但完整的实现
        "decoder": "基础实现",  # 使用基础但完整的实现
        "torch_geometric": "可用" if _HAS_TORCH_GEOMETRIC else "不可用",
        "geometry_extraction": "已启用",  # 新增：几何特征提取状态
        "version": __version__,
        "status": __status__
    }
    return status