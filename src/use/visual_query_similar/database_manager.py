# File: src/use/visual_query_similar/database_manager.py
# 数据库管理模块 / Database management module

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class DatabaseManager:
    """数据库管理器 / Database Manager"""
    
    def __init__(self, db_path: Path):
        """
        初始化数据库管理器 / Initialize database manager
        
        Args:
            db_path: 数据库文件路径 / Database file path
        """
        self.db_path = db_path
        self.db_columns = []
        self._check_database_structure()
    
    def _check_database_structure(self):
        """检查数据库结构，获取实际的列名 / Check database structure and get column names"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 获取表结构 / Get table structure
            cursor.execute("PRAGMA table_info(patterns)")
            columns = cursor.fetchall()
            
            self.db_columns = [col[1] for col in columns]  # 列名在索引1 / Column names at index 1
            print(f"📋 数据库列名: {self.db_columns}")
            
            # 检查关键列是否存在 / Check if required columns exist
            required_columns = ['id', 'sides', 'quality', 'source_obj', 'boundary_vertices']
            missing_columns = [col for col in required_columns if col not in self.db_columns]
            
            if missing_columns:
                print(f"⚠️ 缺少必要列: {missing_columns}")
            else:
                print("✅ 数据库结构检查通过")
            
            conn.close()
            
        except Exception as e:
            print(f"⚠️ 数据库结构检查失败: {e}")
    
    def get_patch_info_with_geometry(self, dataset, db_index: int) -> Optional[Dict[str, Any]]:
        """
        获取数据库中面片的完整信息，包括几何特征
        Get complete patch information from database, including geometry features
        """
        try:
            if not self.db_path.exists():
                return None
                
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # 从数据集获取模式ID / Get pattern ID from dataset
            pattern_data = dataset[db_index]
            pattern_id = getattr(pattern_data, 'pattern_id', None)
            
            if pattern_id is None:
                conn.close()
                return None
                
            pattern_id_value = pattern_id.item() if hasattr(pattern_id, 'item') else pattern_id
            
            # 动态构建查询以适应不同的数据库结构 / Dynamically build query for different database structures
            # 首先检查哪些字段存在 / First check which fields exist
            available_fields = []
            desired_fields = [
                'id', 'edgebreaker_encoding', 'canonical_form', 'sides', 'complexity_score',
                'num_vertices', 'num_faces', 'source_obj', 'quality', 'boundary_vertices',
                'vertex_normals', 'mean_curvatures', 'gaussian_curvatures', 'edge_lengths',
                'edge_curvatures', 'avg_curvature', 'curvature_variance', 
                'total_boundary_length', 'area'
            ]
            
            for field in desired_fields:
                if field in self.db_columns:
                    available_fields.append(field)
                    
            query = f"SELECT {', '.join(available_fields)} FROM patterns WHERE id = ?"
            cursor.execute(query, (pattern_id_value,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return None
                
            # 创建字段映射字典 / Create field mapping dictionary
            field_values = dict(zip(available_fields, row))
            
            # 解析几何数据 / Parse geometry data
            geometry_data = {}
            try:
                json_fields = ['boundary_vertices', 'vertex_normals', 'mean_curvatures', 
                              'gaussian_curvatures', 'edge_lengths', 'edge_curvatures']
                for json_field in json_fields:
                    if json_field in field_values and field_values[json_field]:
                        geometry_data[json_field] = json.loads(field_values[json_field])
            except json.JSONDecodeError:
                geometry_data = {}
            
            # 构建返回字典，只包含数据库中存在的字段 / Build return dict with only existing fields
            result = {}
            field_mapping = {
                'id': 'pattern_id',
                'edgebreaker_encoding': 'edgebreaker_encoding',
                'canonical_form': 'canonical_form',
                'sides': 'sides',
                'complexity_score': 'complexity_score',
                'num_vertices': 'num_vertices',
                'num_faces': 'num_faces',
                'source_obj': 'source_obj',
                'quality': 'quality',
                'avg_curvature': 'avg_curvature',
                'curvature_variance': 'curvature_variance',
                'total_boundary_length': 'total_boundary_length',
                'area': 'area'
            }
            
            for db_field, result_field in field_mapping.items():
                if db_field in field_values:
                    result[result_field] = field_values[db_field]
            
            result['geometry'] = geometry_data
            return result
            
        except Exception as e:
            print(f"❌ 获取面片几何信息失败: {e}")
            return None
    
    def get_patch_info(self, dataset, db_index: int) -> Optional[Dict[str, Any]]:
        """
        获取数据库中面片的基本信息
        Get basic patch information from database
        """
        try:
            pattern_data = dataset[db_index]
            
            info = {
                'pattern_id': getattr(pattern_data, 'pattern_id', 'N/A'),
                'num_sides': getattr(pattern_data, 'num_sides', 'N/A'),
                'quality': getattr(pattern_data, 'quality', 0),
                'source_obj': getattr(pattern_data, 'source_obj', 'unknown'),
                'canonical_form': getattr(pattern_data, 'canonical_form', 'N/A')
            }
            
            # 转换tensor值为普通值 / Convert tensor values to regular values
            for key, value in info.items():
                if hasattr(value, 'item'):
                    info[key] = value.item()
            
            return info
        except Exception as e:
            print(f"❌ 获取面片基本信息失败: {e}")
            return None
    
    def get_database_columns(self) -> List[str]:
        """获取数据库列名 / Get database column names"""
        return self.db_columns