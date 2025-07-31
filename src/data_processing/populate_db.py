import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import sqlite3
import trimesh
import numpy as np
import networkx as nx
from typing import List, Optional, Tuple, Dict
import tqdm
import json
import importlib.util


# 导入ProperPatternEncoder
def _load_encoder():
    """直接从模块文件加载ProperPatternEncoder"""
    spec = importlib.util.spec_from_file_location(
        "proper_encoder",
        Path(__file__).parent / "proper_encoder.py"
    )
    proper_encoder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proper_encoder_module)
    return proper_encoder_module.ProperPatternEncoder


ProperPatternEncoder = _load_encoder()


def setup_database(db_path: Path) -> sqlite3.Connection:
    """
    设置SQLite数据库，创建包含几何特征的patterns表
    """
    # 如果数据库存在，先删除以确保全新开始
    if db_path.exists():
        print(f"找到旧数据库 {db_path}，正在删除...")
        db_path.unlink()

    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建增强的表结构，包含几何特征
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS patterns
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       -- 拓扑信息
                       edgebreaker_encoding
                       TEXT
                       NOT
                       NULL,
                       canonical_form
                       TEXT
                       NOT
                       NULL,
                       sides
                       INTEGER
                       NOT
                       NULL,
                       -- 基本元数据
                       complexity_score
                       REAL
                       DEFAULT
                       0.0,
                       num_vertices
                       INTEGER
                       DEFAULT
                       0,
                       num_faces
                       INTEGER
                       DEFAULT
                       0,
                       source_obj
                       TEXT
                       NOT
                       NULL,
                       quality
                       TEXT
                       NOT
                       NULL,
                       -- 几何特征（JSON格式存储）
                       boundary_vertices
                       TEXT, -- 边界顶点坐标 [[x,y,z], ...]
                       vertex_normals
                       TEXT, -- 顶点法线 [[nx,ny,nz], ...]
                       mean_curvatures
                       TEXT, -- 平均曲率 [c1, c2, ...]
                       gaussian_curvatures
                       TEXT, -- 高斯曲率 [g1, g2, ...]
                       edge_lengths
                       TEXT, -- 边长度 [l1, l2, ...]
                       edge_curvatures
                       TEXT, -- 边曲率 [ec1, ec2, ...]
                       -- 额外的统计信息
                       avg_curvature
                       REAL, -- 平均曲率均值
                       curvature_variance
                       REAL, -- 曲率方差
                       total_boundary_length
                       REAL, -- 边界总长度
                       area
                       REAL, -- 面片面积
                       UNIQUE
                   (
                       canonical_form,
                       sides
                   )
                       );
                   """)

    conn.commit()
    return conn


def extract_random_patch(mesh: trimesh.Trimesh, face_adjacency_graph: nx.Graph,
                         min_faces: int = 10, max_faces: int = 20) -> Optional[List[int]]:
    """使用BFS从网格中提取连通的面片"""
    num_total_faces = len(mesh.faces)
    if num_total_faces < min_faces:
        return None

    # 尝试多次找到合适的面片
    for _ in range(10):
        start_face_idx = np.random.randint(0, num_total_faces)

        q = [start_face_idx]
        visited = {start_face_idx}
        patch_faces = [start_face_idx]

        while q and len(patch_faces) < max_faces:
            current_face = q.pop(0)
            for neighbor in face_adjacency_graph.neighbors(current_face):
                if neighbor not in visited:
                    visited.add(neighbor)
                    patch_faces.append(neighbor)
                    q.append(neighbor)
                    if len(patch_faces) >= max_faces:
                        break

        if min_faces <= len(patch_faces) <= max_faces:
            return patch_faces

    return None


def extract_geometric_features(mesh: trimesh.Trimesh, patch_face_indices: List[int]) -> Optional[Dict]:
    """
    从网格面片中提取几何特征
    """
    try:
        # 获取边界路径
        boundary_path = mesh.outline(patch_face_indices)
        if boundary_path is None or len(boundary_path.entities) == 0:
            return None

        # 获取边界顶点
        boundary_entity = boundary_path.entities[0]
        boundary_vertex_indices = boundary_entity.points

        # 1. 提取顶点位置
        vertex_positions = mesh.vertices[boundary_vertex_indices].tolist()

        # 2. 提取顶点法线
        vertex_normals = mesh.vertex_normals[boundary_vertex_indices].tolist()

        # 3. 计算顶点曲率
        # 计算平均曲率
        mean_curvatures = []
        gaussian_curvatures = []

        # 使用trimesh的曲率计算功能
        radius = mesh.scale / 50.0  # 自适应半径

        for vertex_idx in boundary_vertex_indices:
            # 计算平均曲率
            mean_curv = trimesh.curvature.discrete_mean_curvature_measure(
                mesh, mesh.vertices[[vertex_idx]], radius
            )[0]
            mean_curvatures.append(float(mean_curv))

            # 计算高斯曲率
            gaussian_curv = trimesh.curvature.discrete_gaussian_curvature_measure(
                mesh, mesh.vertices[[vertex_idx]], radius
            )[0]
            gaussian_curvatures.append(float(gaussian_curv))

        # 4. 计算边长度和边曲率
        edge_lengths = []
        edge_curvatures = []

        num_boundary_vertices = len(boundary_vertex_indices)
        for i in range(num_boundary_vertices):
            v1_idx = boundary_vertex_indices[i]
            v2_idx = boundary_vertex_indices[(i + 1) % num_boundary_vertices]

            # 边长度
            edge_length = np.linalg.norm(mesh.vertices[v2_idx] - mesh.vertices[v1_idx])
            edge_lengths.append(float(edge_length))

            # 边曲率（两个端点曲率的平均）
            edge_curv = (mean_curvatures[i] + mean_curvatures[(i + 1) % num_boundary_vertices]) / 2
            edge_curvatures.append(float(edge_curv))

        # 5. 计算统计信息
        avg_curvature = float(np.mean(mean_curvatures))
        curvature_variance = float(np.var(mean_curvatures))
        total_boundary_length = float(np.sum(edge_lengths))

        # 6. 计算面片面积
        patch_faces = mesh.faces[patch_face_indices]
        area = 0.0
        for face in patch_faces:
            # 计算三角形面积（假设是三角网格）
            v0, v1, v2 = mesh.vertices[face]
            area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        return {
            'boundary_vertices': vertex_positions,
            'vertex_normals': vertex_normals,
            'mean_curvatures': mean_curvatures,
            'gaussian_curvatures': gaussian_curvatures,
            'edge_lengths': edge_lengths,
            'edge_curvatures': edge_curvatures,
            'avg_curvature': avg_curvature,
            'curvature_variance': curvature_variance,
            'total_boundary_length': total_boundary_length,
            'area': float(area)
        }

    except Exception as e:
        print(f"提取几何特征失败: {e}")
        return None


def encode_patch_to_pattern(mesh: trimesh.Trimesh, patch_face_indices: List[int]) -> Optional[
    Tuple[str, str, int, Dict, Dict]]:
    """
    编码几何面片，同时提取拓扑和几何特征
    """
    try:
        # 1. 使用多弦折叠算法编码拓扑
        encoder = ProperPatternEncoder()
        encoding_result = encoder.encode_patch_to_pattern(mesh, patch_face_indices)

        if encoding_result is None:
            return None

        edgebreaker_encoding, num_sides = encoding_result

        # 2. 生成规范形式（用于去重）
        canonical_form = edgebreaker_encoding.strip()

        # 3. 计算拓扑元数据
        patch_faces = mesh.faces[patch_face_indices]
        unique_vertices = np.unique(patch_faces)
        num_faces = len(patch_faces)
        num_vertices = len(unique_vertices)
        complexity_score = num_faces / max(num_vertices, 1)

        topology_metadata = {
            'complexity_score': complexity_score,
            'num_vertices': num_vertices,
            'num_faces': num_faces
        }

        # 4. 提取几何特征
        geometric_features = extract_geometric_features(mesh, patch_face_indices)

        if geometric_features is None:
            return None

        return edgebreaker_encoding, canonical_form, num_sides, topology_metadata, geometric_features

    except Exception as e:
        print(f"编码面片失败: {e}")
        return None


def main():
    """
    处理所有.obj文件并填充数据库
    """
    # 设置路径
    model_dir = project_root / "model"
    db_path = project_root / "data" / "raw" / "patches.db"
    patches_per_model = 100

    print(f"设置数据库... (Setting up database at {db_path})")
    conn = setup_database(db_path)
    cursor = conn.cursor()

    print(f"检查模型目录: {model_dir}")
    print(f"模型目录存在: {model_dir.exists()}")

    # 查找所有.obj文件
    obj_files = list(model_dir.glob("**/*.obj"))
    print(f"找到 {len(obj_files)} 个模型文件")

    # 显示找到的文件
    for i, obj_file in enumerate(obj_files[:5]):  # 只显示前5个
        print(f"  {i + 1}: {obj_file.relative_to(model_dir)}")

    # 处理每个模型文件
    for obj_path in tqdm.tqdm(obj_files, desc="处理模型"):
        # 确定质量标签
        quality = obj_path.parent.name
        if quality not in ['new', 'old']:
            tqdm.tqdm.write(f"跳过非预期目录中的文件: {obj_path}")
            continue

        try:
            # 加载网格
            mesh = trimesh.load(obj_path, process=True)
            if not isinstance(mesh, trimesh.Trimesh):
                tqdm.tqdm.write(f"跳过非Trimesh对象: {obj_path.name}")
                continue

            # 预处理网格
            mesh.merge_vertices()
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()

            # 构建面邻接图
            face_adjacency_graph = nx.from_edgelist(mesh.face_adjacency)

        except Exception as e:
            tqdm.tqdm.write(f"加载失败 {obj_path.name}: {e}")
            continue

        # 从每个模型提取多个面片
        successful_patches = 0
        for _ in range(patches_per_model):
            # 提取随机面片
            patch_indices = extract_random_patch(mesh, face_adjacency_graph)
            if not patch_indices:
                continue

            # 编码面片并提取特征
            encoding_result = encode_patch_to_pattern(mesh, patch_indices)
            if not encoding_result:
                continue

            edgebreaker_encoding, canonical_form, sides, topology_metadata, geometric_features = encoding_result

            try:
                # 将几何特征转换为JSON字符串
                cursor.execute("""
                               INSERT INTO patterns (edgebreaker_encoding, canonical_form, sides,
                                                     complexity_score, num_vertices, num_faces,
                                                     source_obj, quality,
                                                     boundary_vertices, vertex_normals,
                                                     mean_curvatures, gaussian_curvatures,
                                                     edge_lengths, edge_curvatures,
                                                     avg_curvature, curvature_variance,
                                                     total_boundary_length, area)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                               """, (
                                   edgebreaker_encoding, canonical_form, sides,
                                   topology_metadata['complexity_score'],
                                   topology_metadata['num_vertices'],
                                   topology_metadata['num_faces'],
                                   obj_path.name, quality,
                                   json.dumps(geometric_features['boundary_vertices']),
                                   json.dumps(geometric_features['vertex_normals']),
                                   json.dumps(geometric_features['mean_curvatures']),
                                   json.dumps(geometric_features['gaussian_curvatures']),
                                   json.dumps(geometric_features['edge_lengths']),
                                   json.dumps(geometric_features['edge_curvatures']),
                                   geometric_features['avg_curvature'],
                                   geometric_features['curvature_variance'],
                                   geometric_features['total_boundary_length'],
                                   geometric_features['area']
                               ))

                successful_patches += 1

            except sqlite3.IntegrityError:
                # 规范形式已存在，跳过
                pass
            except Exception as e:
                tqdm.tqdm.write(f"插入失败 for {obj_path.name}: {e}")

        if successful_patches > 0:
            tqdm.tqdm.write(f"成功从 {obj_path.name} 提取 {successful_patches} 个面片")

    conn.commit()

    # 显示数据库统计信息
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

    conn.close()


if __name__ == '__main__':
    main()