# File: src/common/path_manager.py
# 统一路径管理模块 / Unified path management module

import os
import sys
from pathlib import Path
from typing import Optional, Union

class PathManager:
    """
    统一的路径管理器，确保所有文件都能正确找到项目根目录和相关文件
    """
    
    _instance = None
    _project_root = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._project_root is None:
            # 使用__file__来定位path_manager.py,然后找到项目根目录
            # path_manager.py -> common -> src -> project_root
            self._project_root = Path(__file__).parent.parent.parent.resolve()
            
    @property
    def project_root(self) -> Path:
        """获取项目根目录"""
        return self._project_root
    
    @property
    def src_dir(self) -> Path:
        """源代码目录"""
        return self.project_root / 'src'
    
    @property
    def data_dir(self) -> Path:
        """数据目录"""
        return self.project_root / 'data'
    
    @property
    def data_raw_dir(self) -> Path:
        """原始数据目录"""
        return self.data_dir / 'raw'
    
    @property
    def data_processed_dir(self) -> Path:
        """处理后数据目录"""
        return self.data_dir / 'processed'
    
    @property
    def model_dir(self) -> Path:
        """模型文件目录"""
        return self.project_root / 'model'
    
    @property
    def configs_dir(self) -> Path:
        """配置文件目录"""
        return self.project_root / 'configs'
    
    @property
    def database_path(self) -> Path:
        """数据库文件路径"""
        return self.data_raw_dir / 'patches.db'
    
    @property
    def checkpoint_dir(self) -> Path:
        """检查点目录"""
        # 根据项目结构，检查点在 training 模块内
        return self.project_root / self.configs.get('training', {}).get('checkpoint_dir', 'src/train/training/checkpoints')

    def get_config_path(self, config_name: str = 'config.yaml') -> Path:
        """获取配置文件路径"""
        return self.configs_dir / config_name
    
    def get_model_path(self, model_name: str) -> Path:
        """获取模型文件路径"""
        return self.model_dir / model_name
    
    def get_checkpoint_path(self, checkpoint_name: str) -> Path:
        """获取检查点文件路径"""
        return self.checkpoint_dir / checkpoint_name
    
    def ensure_directory_exists(self, path: Union[str, Path]) -> Path:
        """确保目录存在，如果不存在则创建"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def add_src_to_path(self):
        """将src目录添加到Python路径中"""
        src_path = str(self.src_dir)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            # print(f"✅ 已添加到Python路径: {src_path}")
    
    def validate_project_structure(self) -> bool:
        """验证项目结构是否完整"""
        required_dirs = [
            self.src_dir,
            self.data_dir,
            self.configs_dir,
            self.model_dir
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print("❌ 项目结构不完整，缺少以下目录:")
            for missing_dir in missing_dirs:
                print(f"   • {missing_dir}")
            return False
        
        print("✅ 项目结构验证通过")
        return True
    
    def setup_project_paths(self):
        """设置项目路径环境"""
        # 添加src到Python路径
        self.add_src_to_path()
        
        # 设置环境变量
        os.environ['PROJECT_ROOT'] = str(self.project_root)
        os.environ['DATA_ROOT'] = str(self.data_dir)
        os.environ['MODEL_ROOT'] = str(self.model_dir)
        
        # 确保必要目录存在
        self.ensure_directory_exists(self.data_raw_dir)
        self.ensure_directory_exists(self.data_processed_dir)
        
        # print("🔧 项目路径环境设置完成")
    
    def print_project_info(self):
        """打印项目信息"""
        print("\n" + "="*60)
        print("📁 项目路径信息")
        print("="*60)
        print(f"项目根目录: {self.project_root}")
        print(f"源代码目录: {self.src_dir}")
        print(f"数据目录: {self.data_dir}")
        print(f"模型目录: {self.model_dir}")
        print(f"配置目录: {self.configs_dir}")
        print(f"数据库路径: {self.database_path}")
        # print(f"检查点目录: {self.checkpoint_dir}")
        print("="*60)

# 创建全局路径管理器实例
path_manager = PathManager()

# 便捷函数
def get_project_root() -> Path:
    """获取项目根目录"""
    return path_manager.project_root

def get_database_path() -> Path:
    """获取数据库路径"""
    return path_manager.database_path

def get_config_path(config_name: str = 'config.yaml') -> Path:
    """获取配置文件路径"""
    return path_manager.get_config_path(config_name)

def setup_project_environment():
    """设置项目环境"""
    path_manager.setup_project_paths()
    # path_manager.validate_project_structure()
    return path_manager

# 自动设置路径（当模块被导入时）
if __name__ != "__main__":
    path_manager.add_src_to_path()
