# File: src/use/visual_query_similar/main.py
# 主入口模块 / Main entry module

import sys
from pathlib import Path
from typing import Optional

# 添加项目路径 / Add project path
project_root = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))

# 处理导入问题 / Handle import issues
try:
    # 尝试相对导入（在包内使用时） / Try relative import (when used within package)
    from .config import ConfigManager
    from .database_manager import DatabaseManager
    from .mesh_loader import MeshLoader
    from .visualizer_core import VisualizerCore
    from .html_generator import HTMLGenerator
    from .utils import Utils
except ImportError:
    # 绝对导入（直接运行时） / Absolute import (when run directly)
    sys.path.insert(0, str(Path(__file__).parent))
    from config import ConfigManager
    from database_manager import DatabaseManager
    from mesh_loader import MeshLoader
    from visualizer_core import VisualizerCore
    from html_generator import HTMLGenerator
    from utils import Utils

# 导入项目依赖 / Import project dependencies
from src.train.data_processing.triplet_generator import TripletGenerator
from src.train.data_processing.pyg_dataset import PatchDataset
from src.use.query_core import QueryEngine


class EnhancedMeshPatchVisualizer:
    """增强版网格面片可视化器 / Enhanced Mesh Patch Visualizer"""

    def __init__(self, config_path: str):
        """
        初始化可视化器 / Initialize visualizer
        
        Args:
            config_path: 配置文件路径 / Configuration file path
        """
        print(f"📝 读取配置文件: {config_path}")
        
        # 初始化各个组件 / Initialize components
        self.config_manager = ConfigManager(config_path)
        self.database_manager = DatabaseManager(self.config_manager.get_database_path())
        self.mesh_loader = MeshLoader()
        self.visualizer_core = VisualizerCore()
        self.html_generator = HTMLGenerator()
        
        # 验证路径 / Validate paths
        if not self.config_manager.validate_paths():
            raise FileNotFoundError("必要文件路径验证失败")
        
        print("🔧 初始化查询引擎...")
        # 初始化查询引擎 / Initialize query engine
        self.query_engine = QueryEngine(config_path)
        
        print("📊 加载数据集...")
        # 加载数据集 / Load dataset
        self.dataset = PatchDataset(root=str(self.config_manager.get_data_root()))
        
        print(f"✅ 可视化器初始化完成，数据库包含 {len(self.dataset)} 个面片")

    def create_enhanced_visualization(self, mesh_path: str, max_results: int = 6):
        """
        创建增强版可视化 / Create enhanced visualization
        
        Args:
            mesh_path: 网格文件路径 / Mesh file path
            max_results: 最大结果数 / Maximum results
            
        Returns:
            (主图表, 详细视图, 查询结果) 或 None / (main figure, detail views, results) or None
        """
        print(f"🎯 开始创建增强版可视化: {Path(mesh_path).name}")

        # 1. 提取查询面片 / Extract query patch
        try:
            generator = TripletGenerator(mesh_path, self.dataset)
        except Exception as e:
            print(f"❌ 创建TripletGenerator失败: {e}")
            return None

        # 提取随机面片 / Extract random patch
        query_patch_indices = Utils.extract_random_patch_with_fallback(generator)
        if not query_patch_indices:
            print("❌ 无法提取查询面片")
            return None

        # 2. 创建查询数据 / Create query data
        query_anchor = Utils.create_anchor_with_fallback(generator, query_patch_indices)
        if query_anchor is None:
            print("❌ 无法创建查询锚点")
            return None

        print(f"✅ 查询面片: {query_anchor.num_nodes} 个边界节点")

        # 3. 执行查询 / Execute query
        results = self.query_engine.query(query_anchor, k=max_results)
        if not results:
            print("❌ 未找到相似结果")
            return None

        print(f"✅ 找到 {len(results)} 个相似面片")

        # 4. 创建主要布局（概览视图） / Create main layout (overview)
        main_fig = self.visualizer_core.create_overview_figure(
            mesh_path, query_patch_indices, results, max_results,
            self.database_manager, self.dataset
        )

        # 5. 创建详细视图数据 / Create detail views
        detail_views = self.visualizer_core.create_detail_views(
            mesh_path, query_patch_indices, results, 
            self.database_manager, self.dataset
        )

        return main_fig, detail_views, results

    def create_interactive_dashboard(self, mesh_path: str) -> Optional[tuple]:
        """
        创建交互式仪表板 / Create interactive dashboard
        
        Args:
            mesh_path: 网格文件路径 / Mesh file path
            
        Returns:
            (HTML内容, 查询结果) 或 None / (HTML content, results) or None
        """
        print("🎨 创建交互式仪表板...")

        result = self.create_enhanced_visualization(mesh_path, max_results=6)
        if result is None:
            return None

        main_fig, detail_views, results = result

        # 收集面片信息用于信息面板 / Collect patch info for info panels
        patch_infos = {}
        
        # 查询面片信息（从网格本身生成的基本信息） / Query patch info (basic info from mesh itself)
        patch_infos['query'] = {
            'pattern_id': 'Query',
            'sides': 'N/A',
            'source_obj': Path(mesh_path).name,
            'quality': 'Query',
            'geometry': {}
        }
        
        # 相似面片信息 / Similar patches info
        for idx, (db_index, similarity) in enumerate(results[:5]):
            similar_patch_info = self.database_manager.get_patch_info_with_geometry(self.dataset, db_index)
            if similar_patch_info:
                patch_infos[f'similar_{idx}'] = similar_patch_info

        # 生成HTML内容 / Generate HTML content
        html_content = self.html_generator.generate_dashboard_html(
            main_fig, detail_views, results, mesh_path, patch_infos
        )

        return html_content, results

    def save_interactive_visualization(self, html_content: str, filename: str = "enhanced_patch_comparison.html") -> Path:
        """
        保存交互式可视化结果 / Save interactive visualization results
        
        Args:
            html_content: HTML内容 / HTML content
            filename: 文件名 / Filename
            
        Returns:
            输出文件路径 / Output file path
        """
        output_path = self.config_manager.get_project_root() / "output" / filename
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ 增强版可视化已保存: {output_path}")
        return output_path


def main():
    """主函数 / Main function"""
    print("🎨 增强版3D面片可视化查询系统")
    print("=" * 60)
    
    # 设置环境 / Setup environment
    Utils.setup_environment()
    
    # 验证依赖 / Validate dependencies
    if not Utils.validate_dependencies():
        return
    
    # 查找配置文件 / Find config file
    config_path = Utils.find_config_file(project_root)
    if not config_path:
        return

    # 创建可视化器 / Create visualizer
    print("🚀 开始初始化增强版可视化器...")
    try:
        visualizer = EnhancedMeshPatchVisualizer(config_path)
        print("✅ 可视化器初始化成功")
    except Exception as e:
        print(f"❌ 可视化器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 查找网格文件 / Find mesh files
    model_dir = project_root / 'model'
    mesh_files = MeshLoader.find_mesh_files(model_dir)

    if not mesh_files:
        print("❌ 未找到网格文件")
        return

    # 选择测试网格 / Select test mesh
    try:
        selected_mesh = Utils.select_mesh_interactive(mesh_files, project_root)
    except Exception as e:
        print(f"❌ 网格选择失败: {e}")
        return

    print(f"🎯 选定: {selected_mesh.relative_to(project_root)}")

    # 创建增强版可视化 / Create enhanced visualization
    print("\n🎨 生成增强版可视化...")
    result = visualizer.create_interactive_dashboard(str(selected_mesh))

    if result is not None:
        html_content, results = result

        # 保存结果 / Save results
        output_file = f"enhanced_patch_query_{selected_mesh.stem}.html"
        output_path = visualizer.save_interactive_visualization(html_content, output_file)

        # 打印成功信息 / Print success info
        Utils.print_success_info(output_path)

        # 尝试自动打开 / Try to auto-open
        Utils.try_open_browser(output_path)
    else:
        print("❌ 可视化生成失败")


if __name__ == '__main__':
    print("🚀 脚本开始执行...")
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    except Exception as e:
        print(f"\n💥 程序错误: {e}")
        import traceback
        traceback.print_exc()