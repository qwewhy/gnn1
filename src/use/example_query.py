#!/usr/bin/env python3
# File: src/use/example_query.py
# 查询示例脚本 / Example query script
"""
实用的查询示例脚本
使用训练好的模型进行几何面片相似性检索
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
from pathlib import Path

# 使用统一的路径管理器来设置环境
try:
    from src.common.path_manager import setup_project_environment
except ImportError:
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / 'src'))
    from src.common.path_manager import setup_project_environment

path_manager = setup_project_environment()

# 现在可以安全地导入其他模块
from src.train.data_processing.pyg_dataset import PatchDataset
from src.use.query_core import QueryEngine
import yaml


def find_config_file() -> Path:
    """智能查找配置文件"""
    # 尝试默认的config.yaml
    config_path = path_manager.get_config_path('config.yaml')
    if config_path.exists():
        return config_path
    
    # 如果找不到，尝试在configs目录下查找其他可能的配置文件
    for name in ['advanced_training_config.yaml', 'hard_mining_config.yaml']:
        alt_path = path_manager.get_config_path(name)
        if alt_path.exists():
            print(f"⚠️ 未找到 'config.yaml', 使用备用配置: {name}")
            return alt_path
            
    return None

def find_available_meshes() -> list:
    """查找所有可用的.obj网格文件"""
    return list(path_manager.model_dir.glob('**/*.obj'))

def check_prerequisites(config_path: Path) -> tuple:
    """检查运行所需的所有文件和目录"""
    issues = []
    
    if not config_path:
        issues.append("无法找到任何有效的配置文件。")
        return False, issues
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 1. 检查数据库
    if not path_manager.database_path.exists():
        issues.append(f"数据库文件不存在: {path_manager.database_path}")
        issues.append("  💡 提示: 请先运行 'python -m src.train.data_processing.populate_db' 来生成数据库。")

    # 2. 检查PyG数据集文件
    processed_dataset_path = path_manager.data_processed_dir / 'pyg_patch_dataset_with_geometry.pt'
    if not processed_dataset_path.exists():
        issues.append(f"预处理数据集不存在: {processed_dataset_path}")
        issues.append("  💡 提示: 数据集将在首次训练或查询时自动生成，但数据库必须存在。")
        
    # 3. 检查模型检查点
    checkpoint_dir = path_manager.project_root / config['training']['checkpoint_dir']
    best_model_path = checkpoint_dir / 'best_model.pt'
    if not best_model_path.exists():
        issues.append(f"训练好的模型不存在: {best_model_path}")
        issues.append("  💡 提示: 请先运行训练脚本 'python -m src.train.training.improved_train'。")

    # 4. 检查索引文件 (如果QueryEngine需要的话)
    index_path = path_manager.data_processed_dir / 'pattern_index.pt'
    if not index_path.exists():
        issues.append(f"模式索引文件不存在: {index_path}")
        issues.append("  💡 提示: 请运行 'python -m src.use.build_index' 来创建索引。")

    return len(issues) == 0, issues


def display_results_table(results: list, dataset: PatchDataset):
    """表格形式显示结果"""
    if not results:
        print("❌ 没有找到相似的结果")
        return

    print(f"\n🎯 找到 {len(results)} 个相似的拓扑模式:")
    headers = ["排名", "DB索引", "模式ID", "边数", "质量", "相似度", "几何?"]
    print("-" * 60)
    print(f"{headers[0]:<5}{headers[1]:<8}{headers[2]:<10}{headers[3]:<7}{headers[4]:<8}{headers[5]:<10}{headers[6]:<7}")
    print("=" * 60)

    for rank, (db_index, similarity) in enumerate(results, 1):
        try:
            pattern_data = dataset[db_index]
            
            pattern_id = getattr(pattern_data, 'pattern_id', 'N/A')
            pattern_id = pattern_id.item() if hasattr(pattern_id, 'item') else pattern_id
            
            num_sides = getattr(pattern_data, 'num_sides', 'N/A')
            num_sides = num_sides.item() if hasattr(num_sides, 'item') else num_sides

            quality_val = getattr(pattern_data, 'quality', 0)
            quality_val = quality_val.item() if hasattr(quality_val, 'item') else quality_val
            quality = 'new✨' if quality_val == 1 else 'old⚠️'
            
            has_geom = '✅' if getattr(pattern_data, 'has_geometry', False) else '❌'

            print(f"{rank:<5}{db_index:<8}{pattern_id:<10}{num_sides:<7}{quality:<8}{similarity:<10.4f}{has_geom:<7}")

        except Exception:
            print(f"{rank:<5}{db_index:<8}{'获取错误':<10}")
    print("-" * 60)

def main():
    """主函数"""
    print("🚀 几何面片相似性检索系统")
    print("=" * 60)
    path_manager.print_project_info()

    # 1. 查找并验证配置文件
    config_path = find_config_file()
    
    # 2. 检查所有前提条件
    prerequisites_ok, issues = check_prerequisites(config_path)
    if not prerequisites_ok:
        print("\n❌ 系统检查失败，缺少必要文件:")
        for issue in issues:
            print(f"   • {issue}")
        return

    print("\n✅ 系统检查通过，所有文件就绪！")

    # 3. 选择查询网格
    available_meshes = find_available_meshes()
    if not available_meshes:
        print("❌ 在 'model' 目录下未找到任何 .obj 网格文件。")
        return
        
    print("\n📋 请选择一个用于查询的网格文件:")
    for i, mesh_file in enumerate(available_meshes):
        print(f"   {i + 1}. {mesh_file.relative_to(path_manager.project_root)}")
    
    try:
        choice = int(input(f"请输入编号 (1-{len(available_meshes)}), 回车默认选1: ") or 1) - 1
        selected_mesh_path = available_meshes[choice]
    except (ValueError, IndexError):
        print("无效选择，自动选择第一个。")
        selected_mesh_path = available_meshes[0]

    print(f"🎯 选定查询网格: {selected_mesh_path.name}")

    # 4. 初始化查询引擎
    try:
        query_engine = QueryEngine(str(config_path))
    except Exception as e:
        print(f"❌ 初始化查询引擎失败: {e}")
        return

    # 5. 从数据集中提取一个随机面片作为查询目标
    try:
        dataset = PatchDataset() # 使用默认路径
        if len(dataset) == 0:
            print("❌ 数据集为空，无法提取查询面片。")
            return

        # 随机选择一个样本作为查询锚点
        import random
        random_index = random.randint(0, len(dataset) - 1)
        query_anchor = dataset[random_index]
        
        print(f"✅ 成功从数据集中提取一个随机查询面片 (DB 索引: {random_index}, {query_anchor.num_nodes} 个节点)。")
    except Exception as e:
        print(f"❌ 提取查询面片失败: {e}")
        return

    # 6. 执行查询
    print("\n🔍 正在执行相似性搜索...")
    results = query_engine.query(query_anchor, k=10)

    # 7. 显示结果
    display_results_table(results, dataset)
    print("\n🎉 查询完成!")

if __name__ == '__main__':
    main()
