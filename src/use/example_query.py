#!/usr/bin/env python3
# File: src/use/example_query.py
# 查询示例脚本 / Example query script
"""
实用的查询示例脚本
使用训练好的模型进行几何面片相似性检索
"""

# 🔧 修复OpenMP库冲突问题 - 必须在导入其他库之前设置
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # 限制OpenMP线程数，提高稳定性

from pathlib import Path
import yaml
import sys

# 确保可以从src导入模块 - 修复路径问题
project_root = Path(__file__).parent.parent.parent.absolute()  # 获取项目根目录
sys.path.insert(0, str(project_root))  # 使用insert确保优先级

# 切换到项目根目录，确保相对路径正确
os.chdir(project_root)
print(f"🏠 项目根目录: {project_root}")
print(f"🔧 OpenMP设置: KMP_DUPLICATE_LIB_OK={os.environ.get('KMP_DUPLICATE_LIB_OK', 'None')}")

from src.train.data_processing.triplet_generator import TripletGenerator
from src.train.data_processing.pyg_dataset import PatchDataset
from src.use.query import QueryEngine  # 修正导入路径


def find_config_file() -> Path:
    """智能查找配置文件"""
    possible_paths = [
        Path('configs/config.yaml'),
        Path('configs') / 'config.yaml',
        project_root / 'configs' / 'config.yaml',
        Path(__file__).parent / 'configs' / 'config.yaml'
    ]

    for config_path in possible_paths:
        if config_path.exists():
            print(f"✅ 找到配置文件: {config_path.absolute()}")
            return config_path

    print("❌ 未找到配置文件，尝试过的路径:")
    for path in possible_paths:
        print(f"   • {path.absolute()}")

    return None


def find_available_meshes(project_root: Path) -> list:
    """查找可用的网格文件"""
    possible_dirs = ['model', 'models', 'data/meshes', 'meshes']
    mesh_files = []

    print(f"🔍 搜索网格文件目录:")
    for dir_name in possible_dirs:
        mesh_dir = project_root / dir_name
        print(f"   检查: {mesh_dir.absolute()} - {'存在' if mesh_dir.exists() else '不存在'}")
        if mesh_dir.exists():
            found_files = list(mesh_dir.glob('**/*.obj'))  # 递归搜索
            mesh_files.extend(found_files)
            print(f"      找到 {len(found_files)} 个.obj文件")

    return mesh_files


def check_prerequisites(config_path: Path) -> tuple:
    """检查运行前提条件"""
    issues = []

    # 检查配置文件
    if not config_path.exists():
        issues.append(f"配置文件不存在: {config_path.absolute()}")
        return False, issues

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        issues.append(f"配置文件读取失败: {e}")
        return False, issues

    # 检查数据目录
    data_root = Path(config['data']['root'])
    if not data_root.exists():
        issues.append(f"数据目录不存在: {data_root.absolute()}")

    # 检查数据库文件
    db_path = data_root / 'raw' / 'patches.db'
    if not db_path.exists():
        issues.append(f"数据库文件不存在: {db_path.absolute()}")
        issues.append("提示: 运行 python -m src.data_processing.populate_db 生成数据库")

    # 检查训练模型
    checkpoint_dir = project_root / config['training']['checkpoint_dir']
    best_model_path = checkpoint_dir / 'best_model.pt'
    if not best_model_path.exists():
        issues.append(f"训练模型不存在: {best_model_path.absolute()}")
        issues.append("提示: 运行训练脚本生成模型，或检查checkpoint_dir路径")

    # 检查模式索引
    index_path = data_root / 'processed' / 'pattern_index.pt'
    if not index_path.exists():
        issues.append(f"模式索引不存在: {index_path.absolute()}")
        issues.append("提示: 运行 python -m src.build_index --config <config_path>")

    return len(issues) == 0, issues


def display_results(results: list, dataset: PatchDataset):
    """修复后的格式化显示查询结果"""
    if not results:
        print("❌ 没有找到相似的结果")
        return

    print(f"\n🎯 找到 {len(results)} 个相似的拓扑模式:")
    print("=" * 70)

    for rank, (db_index, similarity) in enumerate(results, 1):
        try:
            pattern_data = dataset[db_index]

            # 安全地获取属性值并处理tensor
            pattern_id_raw = getattr(pattern_data, 'pattern_id', 'N/A')
            pattern_id = pattern_id_raw.item() if hasattr(pattern_id_raw, 'item') else pattern_id_raw
            
            num_sides_raw = getattr(pattern_data, 'num_sides', 'N/A')
            num_sides = num_sides_raw.item() if hasattr(num_sides_raw, 'item') else num_sides_raw
            
            quality_val_raw = getattr(pattern_data, 'quality', 0)
            quality_val = quality_val_raw.item() if hasattr(quality_val_raw, 'item') else quality_val_raw
            quality = 'new✨' if quality_val == 1 else 'old⚠️'
            
            complexity_raw = getattr(pattern_data, 'complexity_score', 'N/A')
            complexity = complexity_raw.item() if hasattr(complexity_raw, 'item') else complexity_raw

            # 获取来源信息
            canonical_form = getattr(pattern_data, 'canonical_form', 'N/A')
            if canonical_form != 'N/A' and len(str(canonical_form)) > 20:
                source = str(canonical_form)[:20] + '...'
            else:
                source = str(canonical_form)

            print(f"🏆 排名 {rank}:")
            print(f"   📊 数据库索引: {db_index}")
            print(f"   🆔 模式ID: {pattern_id}")
            print(f"   🔢 边界边数: {num_sides}")
            print(f"   ⭐ 质量标签: {quality}")
            print(f"   📈 相似度分数: {similarity:.4f}")
            print(f"   🧮 复杂度: {complexity}")
            print(f"   📝 规范形式: {source}")

            # 尝试显示几何特征信息
            if hasattr(pattern_data, 'has_geometry') and pattern_data.has_geometry:
                print(f"   🎨 包含几何特征: ✅")
            else:
                print(f"   🎨 包含几何特征: ❌")

            print("-" * 50)

        except Exception as e:
            print(f"🏆 排名 {rank}:")
            print(f"   📊 数据库索引: {db_index}")
            print(f"   ❌ 数据获取错误: {str(e)}")
            print("-" * 50)


def display_results_table(results: list, dataset: PatchDataset):
    """表格形式显示结果（备用方案）"""
    if not results:
        print("❌ 没有找到相似的结果")
        return

    print(f"\n🎯 找到 {len(results)} 个相似的拓扑模式:")
    print("=" * 80)

    # 表头
    headers = ["排名", "DB索引", "模式ID", "边数", "质量", "相似度", "复杂度"]
    header_line = ""
    for header in headers:
        header_line += header.ljust(10)
    print(header_line)
    print("-" * 80)

    for rank, (db_index, similarity) in enumerate(results, 1):
        try:
            pattern_data = dataset[db_index]

            # 获取数据并正确处理tensor
            pattern_id_raw = getattr(pattern_data, 'pattern_id', 'N/A')
            pattern_id = str(pattern_id_raw.item() if hasattr(pattern_id_raw, 'item') else pattern_id_raw)[:8]
            
            num_sides_raw = getattr(pattern_data, 'num_sides', 'N/A')
            num_sides = str(num_sides_raw.item() if hasattr(num_sides_raw, 'item') else num_sides_raw)
            
            quality_val_raw = getattr(pattern_data, 'quality', 0)
            quality_val = quality_val_raw.item() if hasattr(quality_val_raw, 'item') else quality_val_raw
            quality = 'new' if quality_val == 1 else 'old'
            
            sim_str = f"{similarity:.3f}"
            
            complexity_raw = getattr(pattern_data, 'complexity_score', 0)
            complexity = complexity_raw.item() if hasattr(complexity_raw, 'item') else complexity_raw
            comp_str = f"{complexity:.2f}" if complexity != 'N/A' else 'N/A'

            # 构建行
            row_data = [str(rank), str(db_index), pattern_id, num_sides, quality, sim_str, comp_str]
            row_line = ""
            for data in row_data:
                row_line += data.ljust(10)
            print(row_line)

        except Exception as e:
            error_row = [str(rank), str(db_index), "ERROR", "N/A", "N/A", "N/A", "N/A"]
            row_line = ""
            for data in error_row:
                row_line += data.ljust(10)
            print(row_line)


def main():
    """主函数"""
    print("🚀 几何面片相似性检索系统")
    print("=" * 60)
    print(f"📍 当前工作目录: {Path.cwd()}")

    # 智能查找配置文件
    print("\n🔍 查找配置文件...")
    config_path = find_config_file()

    if config_path is None:
        print("❌ 无法找到配置文件 config.yaml")
        print("💡 请确保配置文件位于 configs/ 目录下")
        return

    # 检查前提条件
    print("\n🔍 检查系统前提条件...")
    prerequisites_ok, issues = check_prerequisites(config_path)

    if not prerequisites_ok:
        print("❌ 系统检查失败:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 请先解决上述问题后再运行此脚本")
        return

    print("✅ 系统检查通过")

    # 查找可用的网格文件
    print("\n🔍 查找可用的网格文件...")
    available_meshes = find_available_meshes(project_root)

    if not available_meshes:
        print("❌ 未找到任何.obj网格文件")
        print("💡 请确保您的网格文件位于以下目录之一:")
        print("   • model/ (推荐)")
        print("   • models/")
        print("   • data/meshes/")
        print("   • meshes/")
        return

    print(f"✅ 找到 {len(available_meshes)} 个网格文件:")
    for i, mesh_file in enumerate(available_meshes):
        rel_path = mesh_file.relative_to(project_root)
        print(f"   {i + 1}. {rel_path}")

    # 选择测试网格
    if len(available_meshes) == 1:
        selected_mesh = available_meshes[0]
        print(f"\n📋 自动选择唯一网格: {selected_mesh.name}")
    else:
        print(f"\n📋 选择要用于查询的网格文件 (1-{len(available_meshes)}):")
        try:
            choice = int(input("请输入编号 (直接回车使用第1个): ") or "1") - 1
            if 0 <= choice < len(available_meshes):
                selected_mesh = available_meshes[choice]
            else:
                print("❌ 无效选择，使用第一个网格文件")
                selected_mesh = available_meshes[0]
        except (ValueError, KeyboardInterrupt):
            print("❌ 使用第一个网格文件")
            selected_mesh = available_meshes[0]

    print(f"🎯 选定网格: {selected_mesh.relative_to(project_root)}")

    # 初始化查询引擎
    print(f"\n🔧 初始化查询引擎...")
    try:
        query_engine = QueryEngine(str(config_path))
        print("✅ 查询引擎初始化成功")
    except Exception as e:
        print(f"❌ 查询引擎初始化失败: {e}")
        print(f"   详细错误信息: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return

    # 从选中的网格中提取查询面片
    print(f"\n🎯 从网格中提取查询面片...")

    try:
        # 加载数据集
        print("   正在加载数据集...")
        patch_dataset = PatchDataset(root='data')
        print(f"   ✅ 数据集加载成功，包含 {len(patch_dataset)} 个模式")

        # 创建三元组生成器
        print("   正在创建三元组生成器...")
        generator = TripletGenerator(str(selected_mesh), patch_dataset)

        # 尝试提取有效面片
        print("   正在提取查询面片...")
        query_anchor = None
        for attempt in range(10):
            patch_indices = generator.extract_random_patch()
            if patch_indices:
                query_anchor = generator._create_anchor_from_patch(patch_indices)
                if query_anchor is not None:
                    break
            print(f"   尝试 {attempt + 1}/10...")

        if query_anchor is None:
            print("❌ 无法从选定网格中提取有效查询面片")
            return

        print(f"✅ 成功提取查询面片: {query_anchor.num_nodes} 个边界节点")

        # 显示查询面片的详细信息
        print(f"📊 查询面片信息:")
        print(f"   🔢 节点数量: {query_anchor.num_nodes}")
        print(f"   🔗 边数量: {query_anchor.edge_index.shape[1] // 2}")  # 除以2因为是无向图
        print(f"   📏 特征维度: {query_anchor.x.shape[1]}")
        if hasattr(query_anchor, 'edge_attr') and query_anchor.edge_attr is not None:
            print(f"   🔗 边特征维度: {query_anchor.edge_attr.shape[1]}")

    except Exception as e:
        print(f"❌ 面片提取失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 执行查询
    print(f"\n🔍 执行相似性搜索...")
    try:
        k = 10  # 查找前10个最相似的结果
        results = query_engine.query(query_anchor, k=k)

        if results:
            print(f"✅ 搜索完成，找到 {len(results)} 个结果")

            # 使用修复后的显示函数
            display_results(results, patch_dataset)

            # 可选：也显示表格形式
            print(f"\n📋 表格形式总览:")
            display_results_table(results, patch_dataset)

        else:
            print("❌ 搜索完成，但未找到任何结果")

    except Exception as e:
        print(f"❌ 查询执行失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n🎉 查询完成!")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n💥 程序异常退出: {e}")
        import traceback

        traceback.print_exc()