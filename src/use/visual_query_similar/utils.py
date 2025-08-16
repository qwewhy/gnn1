# File: src/use/visual_query_similar/utils.py
# 工具函数模块 / Utility functions module

import os
from pathlib import Path
from typing import List, Optional, Tuple, Any


class Utils:
    """工具函数类 / Utility functions class"""
    
    @staticmethod
    def setup_environment():
        """设置环境变量 / Setup environment variables"""
        # 解决OpenMP问题 / Fix OpenMP issue
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    @staticmethod
    def find_config_file(project_root: Path) -> Optional[str]:
        """
        查找配置文件 / Find configuration file
        
        Args:
            project_root: 项目根目录 / Project root directory
            
        Returns:
            配置文件路径或None / Config file path or None
        """
        possible_configs = [
            'configs/config.yaml',
            project_root / 'configs' / 'config.yaml'
        ]
        
        for path in possible_configs:
            if Path(path).exists():
                print(f"✅ 找到配置文件: {path}")
                return str(path)
        
        print("❌ 未找到配置文件")
        return None
    
    @staticmethod
    def extract_random_patch_with_fallback(generator, mesh_index: int = 0) -> Optional[List[Any]]:
        """
        尝试不同方法提取随机面片 / Try different methods to extract random patch
        
        Args:
            generator: 三元组生成器 / Triplet generator
            mesh_index: 网格索引 / Mesh index
            
        Returns:
            面片索引列表或None / Patch indices list or None
        """
        query_patch_indices = None
        
        # 尝试不同的方法调用 / Try different method calls
        try:
            # 首先尝试带mesh_index参数的调用
            query_patch_indices = generator.extract_random_patch(mesh_index)
        except TypeError:
            try:
                # 如果失败，尝试关键字参数
                query_patch_indices = generator.extract_random_patch(mesh_index=mesh_index)
            except:
                try:
                    # 最后尝试无参数调用
                    query_patch_indices = generator.extract_random_patch()
                except Exception as e:
                    print(f"❌ 无法调用extract_random_patch: {e}")
                    return None
        except Exception as e:
            print(f"❌ extract_random_patch调用失败: {e}")
            return None
        
        return query_patch_indices
    
    @staticmethod
    def create_anchor_with_fallback(generator, patch_indices: List[Any], mesh_index: int = 0) -> Optional[Any]:
        """
        尝试不同方法创建锚点 / Try different methods to create anchor
        
        Args:
            generator: 三元组生成器 / Triplet generator
            patch_indices: 面片索引 / Patch indices
            mesh_index: 网格索引 / Mesh index
            
        Returns:
            锚点对象或None / Anchor object or None
        """
        query_anchor = None
        
        try:
            # 首先尝试带mesh_index参数的调用
            query_anchor = generator._create_anchor_from_patch(patch_indices, mesh_index)
        except TypeError:
            try:
                # 如果失败，尝试关键字参数
                query_anchor = generator._create_anchor_from_patch(patch_indices, mesh_index=mesh_index)
            except:
                try:
                    # 最后尝试无mesh_index参数调用
                    query_anchor = generator._create_anchor_from_patch(patch_indices)
                except Exception as e:
                    print(f"❌ 创建查询锚点失败: {e}")
                    return None
        except Exception as e:
            print(f"❌ 创建查询锚点失败: {e}")
            return None
        
        return query_anchor
    
    @staticmethod
    def select_mesh_interactive(mesh_files: List[Path], project_root: Path) -> Path:
        """
        交互式选择网格文件 / Interactive mesh file selection
        
        Args:
            mesh_files: 网格文件列表 / Mesh files list
            project_root: 项目根目录 / Project root directory
            
        Returns:
            选择的网格文件路径 / Selected mesh file path
        """
        if not mesh_files:
            raise ValueError("没有找到网格文件")
        
        print(f"✅ 找到 {len(mesh_files)} 个网格文件:")
        for i, mesh_file in enumerate(mesh_files):
            rel_path = mesh_file.relative_to(project_root)
            print(f"   {i + 1}. {rel_path}")
        
        try:
            choice = input(f"\n选择网格文件 (1-{len(mesh_files)}, 回车默认第1个): ")
            if choice.strip():
                selected_mesh = mesh_files[int(choice) - 1]
            else:
                selected_mesh = mesh_files[0]
        except (ValueError, IndexError):
            selected_mesh = mesh_files[0]
        
        return selected_mesh
    
    @staticmethod
    def try_open_browser(file_path: Path):
        """
        尝试在浏览器中打开文件 / Try to open file in browser
        
        Args:
            file_path: 文件路径 / File path
        """
        try:
            import webbrowser
            webbrowser.open(f'file://{file_path}')
            print("🌐 已自动打开浏览器")
        except:
            print("💡 请手动打开上述HTML文件")
    
    @staticmethod
    def print_success_info(output_path: Path):
        """
        打印成功信息 / Print success information
        
        Args:
            output_path: 输出文件路径 / Output file path
        """
        print(f"\n🎉 增强版可视化完成！")
        print(f"📂 请打开文件查看: {output_path}")
        print("💡 新功能：")
        print("   • 点击按钮放大查看详细视图")
        print("   • 最多同时显示2个放大视图")
        print("   • 优化的网格线显示")
        print("   • 交互式统计信息")
    
    @staticmethod
    def validate_dependencies():
        """
        验证依赖项 / Validate dependencies
        
        Returns:
            是否所有依赖都可用 / Whether all dependencies are available
        """
        required_modules = [
            'numpy', 'trimesh', 'matplotlib', 'plotly', 
            'yaml', 'sqlite3', 'json'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"❌ 缺少以下依赖: {missing_modules}")
            return False
        
        print("✅ 所有依赖项检查通过")
        return True