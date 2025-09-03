# File: src/train/data_processing/database/database_setup.py
# 数据库设置模块 / Database setup module

import sqlite3
from pathlib import Path
from typing import Optional

def setup_database(db_path: Path) -> sqlite3.Connection:
    """
    设置SQLite数据库，创建包含几何特征的patterns表
    Setup SQLite database with patterns table including geometric features
    """
    # 如果数据库存在，先删除以确保全新开始
    if db_path.exists():
        print(f"找到旧数据库 {db_path}，正在删除...")
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建增强的表结构，包含几何特征
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS patterns
                   (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       -- 拓扑信息
                       edgebreaker_encoding TEXT NOT NULL,
                       canonical_form TEXT NOT NULL,
                       sides INTEGER NOT NULL,
                       -- 基本元数据
                       complexity_score REAL DEFAULT 0.0,
                       num_vertices INTEGER DEFAULT 0,
                       num_faces INTEGER DEFAULT 0,
                       source_obj TEXT NOT NULL,
                       quality TEXT NOT NULL,
                       -- 几何特征（JSON格式存储）
                       boundary_vertices TEXT, -- 边界顶点坐标 [[x,y,z], ...]
                       vertex_normals TEXT, -- 顶点法线 [[nx,ny,nz], ...]
                       mean_curvatures TEXT, -- 平均曲率 [c1, c2, ...]
                       gaussian_curvatures TEXT, -- 高斯曲率 [g1, g2, ...]
                       edge_lengths TEXT, -- 边长度 [l1, l2, ...]
                       edge_curvatures TEXT, -- 边曲率 [ec1, ec2, ...]
                       -- 额外的统计信息
                       avg_curvature REAL, -- 平均曲率均值
                       curvature_variance REAL, -- 曲率方差
                       total_boundary_length REAL, -- 边界总长度
                       area REAL, -- 面片面积
                       ordered_boundary_vertex_indices TEXT, -- 有序边界顶点索引 [idx1, idx2, ...]
                       UNIQUE(canonical_form, sides)
                   );
                   """)

    conn.commit()
    return conn


def get_connection(db_path: Path) -> Optional[sqlite3.Connection]:
    """
    获取数据库连接
    Get database connection
    """
    try:
        if not db_path.exists():
            return None
        return sqlite3.connect(db_path)
    except Exception as e:
        print(f"连接数据库失败: {e}")
        return None
