# File: src/train/data_processing/orchestration/database_populator.py
# 数据库填充协调器 / Database populator orchestrator

import sys
import sqlite3
import trimesh
import json
from pathlib import Path
from typing import Optional
import tqdm
import logging

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

from ..database.database_setup import setup_database
from ..mesh_processing.extraction.mesh_loader import MeshLoader
from .patch_processor import PatchProcessor


class DatabasePopulator:
    """数据库填充协调器 - 统一管理整个数据库填充流程"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mesh_loader = MeshLoader()
        self.patch_processor = PatchProcessor()
    
    def populate_database(self, patches_per_model: int = 100) -> None:
        """
        处理所有.obj文件并填充数据库
        Process all .obj files and populate database
        """
        # 使用统一的路径管理
        db_path = get_database_path()
        model_dir = path_manager.model_dir

        print(f"🗄️ 设置数据库: {db_path}")
        print(f"📁 模型目录: {model_dir}")
        
        conn = setup_database(db_path)
        cursor = conn.cursor()

        obj_files = list(model_dir.glob("**/*.obj"))
        print(f"找到 {len(obj_files)} 个模型文件")

        for obj_path in tqdm.tqdm(obj_files, desc="处理模型"):
            quality = obj_path.parent.name
            if quality not in ['new', 'old']:
                tqdm.tqdm.write(f"跳过非预期目录中的文件: {obj_path}")
                continue

            # 1. 加载和预处理网格
            mesh_data = self.mesh_loader.load_and_preprocess(obj_path)
            if mesh_data is None:
                continue
                
            mesh, face_adjacency_graph = mesh_data

            # 2. 处理面片
            successful_patches = 0
            for _ in range(patches_per_model):
                # 提取和编码面片
                encoding_result = self.patch_processor.extract_and_encode_patch(
                    mesh, face_adjacency_graph
                )
                
                if encoding_result is None:
                    continue

                edgebreaker_encoding, canonical_form, sides, topology_metadata, geometric_features = encoding_result

                # 3. 插入数据库
                if self._insert_to_database(cursor, obj_path.name, quality, 
                                          edgebreaker_encoding, canonical_form, sides,
                                          topology_metadata, geometric_features):
                    successful_patches += 1

            if successful_patches > 0:
                tqdm.tqdm.write(f"成功从 {obj_path.name} 提取 {successful_patches} 个面片")

        # 4. 提交并显示统计信息
        conn.commit()
        self._show_statistics(cursor)
        conn.close()
    
    def _insert_to_database(self, cursor: sqlite3.Cursor, obj_name: str, quality: str,
                           edgebreaker_encoding: str, canonical_form: str, sides: int,
                           topology_metadata: dict, geometric_features: dict) -> bool:
        """插入数据到数据库"""
        try:
            cursor.execute("""
                           INSERT INTO patterns (edgebreaker_encoding, canonical_form, sides,
                                                 complexity_score, num_vertices, num_faces,
                                                 source_obj, quality,
                                                 boundary_vertices, vertex_normals,
                                                 mean_curvatures, gaussian_curvatures,
                                                 edge_lengths, edge_curvatures,
                                                 avg_curvature, curvature_variance,
                                                 total_boundary_length, area,
                                                 ordered_boundary_vertex_indices)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                           """, (
                               edgebreaker_encoding, canonical_form, sides,
                               topology_metadata['complexity_score'],
                               topology_metadata['num_vertices'],
                               topology_metadata['num_faces'],
                               obj_name, quality,
                               json.dumps(geometric_features['boundary_vertices']),
                               json.dumps(geometric_features['vertex_normals']),
                               json.dumps(geometric_features['mean_curvatures']),
                               json.dumps(geometric_features['gaussian_curvatures']),
                               json.dumps(geometric_features['edge_lengths']),
                               json.dumps(geometric_features['edge_curvatures']),
                               geometric_features['avg_curvature'],
                               geometric_features['curvature_variance'],
                               geometric_features['total_boundary_length'],
                               geometric_features['area'],
                               json.dumps(geometric_features['ordered_boundary_vertex_indices'])
                           ))
            return True
            
        except sqlite3.IntegrityError:
            return False  # 重复数据，跳过
        except Exception as e:
            self.logger.error(f"插入失败 for {obj_name}: {e}")
            return False
    
    def _show_statistics(self, cursor: sqlite3.Cursor) -> None:
        """显示数据库统计信息"""
        cursor.execute("SELECT COUNT(*) FROM patterns")
        total_patterns = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM patterns WHERE quality='new'")
        new_patterns = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM patterns WHERE quality='old'")
        old_patterns = cursor.fetchone()[0]

        print(f"\n数据库填充完成！")
        print(f"总模式数: {total_patterns}")
        print(f"'new'模式数: {new_patterns}")
        print(f"'old'模式数: {old_patterns}")


def main():
    """主函数 - 向后兼容"""
    populator = DatabasePopulator()
    populator.populate_database()


if __name__ == '__main__':
    main()
