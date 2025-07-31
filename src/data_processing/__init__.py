"""
数据处理模块 - 完整实现版本（含几何特征提取）
Data processing module - Complete implementation with geometric feature extraction
"""

# 导入核心编码/解码器
try:
    from .proper_encoder_fixed import ProperPatternEncoder, MultiChordFolder, HalfEdgeStructure
    from .proper_decoder_fixed import ProperPatternParser, EdgebreakerDecoder

    print("使用完善的编码/解码器实现")
except ImportError as e:
    print(f"完善实现导入失败，回退到基础实现: {e}")
    from .proper_encoder import ProperPatternEncoder
    from .proper_decoder import ProperPatternParser

# 导入增强的数据库操作（包含几何特征提取）
from .populate_db import (
    setup_database,
    extract_random_patch,
    encode_patch_to_pattern,
    extract_geometric_features  # 新增：几何特征提取
)

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
    # 核心编码/解码
    'ProperPatternEncoder',
    'ProperPatternParser',

    # 数据库操作（增强版）
    'setup_database',
    'extract_random_patch',
    'encode_patch_to_pattern',
    'extract_geometric_features',  # 新增导出
]

# 有条件添加PyG模块
if _HAS_TORCH_GEOMETRIC:
    __all__.extend([
        'PatchDataset',
        'TripletGenerator',
    ])

# 版本信息
__version__ = "1.1.0"  # 升级版本号
__status__ = "Production with Geometry" if _HAS_TORCH_GEOMETRIC else "Limited"


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
        "encoder": "完整实现" if 'MultiChordFolder' in globals() else "基础实现",
        "decoder": "完整实现" if 'EdgebreakerDecoder' in globals() else "基础实现",
        "torch_geometric": "可用" if _HAS_TORCH_GEOMETRIC else "不可用",
        "geometry_extraction": "已启用",  # 新增：几何特征提取状态
        "version": __version__,
        "status": __status__
    }
    return status