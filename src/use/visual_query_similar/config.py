# File: src/use/visual_query_similar/config.py
# 配置管理模块 / Configuration management module

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """配置管理器 / Configuration Manager"""
    
    def __init__(self, config_path: str):
        """
        初始化配置管理器 / Initialize configuration manager
        
        Args:
            config_path: 配置文件路径 / Config file path
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.project_root = self._get_project_root()
        self.data_root = self.project_root / 'data'
        self.db_path = self.data_root / 'raw' / 'patches.db'
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件 / Load configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
            raise
    
    def _get_project_root(self) -> Path:
        """获取项目根目录 / Get project root directory"""
        # 从当前文件路径推导项目根目录
        current_file = Path(__file__).absolute()
        # visual_query_similar -> use -> src -> project_root
        return current_file.parent.parent.parent.parent
    
    def get_database_path(self) -> Path:
        """获取数据库路径 / Get database path"""
        return self.db_path
        
    def get_data_root(self) -> Path:
        """获取数据根目录 / Get data root directory"""
        return self.data_root
        
    def get_project_root(self) -> Path:
        """获取项目根目录 / Get project root directory"""
        return self.project_root
        
    def get_config(self) -> Dict[str, Any]:
        """获取配置字典 / Get configuration dictionary"""
        return self.config
        
    def validate_paths(self) -> bool:
        """验证路径是否存在 / Validate if paths exist"""
        if not self.db_path.exists():
            print(f"❌ 数据库文件不存在: {self.db_path}")
            return False
        if not self.data_root.exists():
            print(f"❌ 数据目录不存在: {self.data_root}")
            return False
        print("✅ 路径验证通过")
        return True