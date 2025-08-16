#!/usr/bin/env python
# File: src/use/visual_query_similar/run_visualizer.py
# 独立启动脚本 / Independent startup script

"""
独立的可视化系统启动脚本 / Independent visualization system startup script

这个脚本可以直接运行，无需担心相对导入问题。
This script can be run directly without worrying about relative import issues.

使用方法 / Usage:
    python run_visualizer.py
    
或者从项目根目录运行 / Or run from project root:
    python src/use/visual_query_similar/run_visualizer.py
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """设置环境和路径 / Setup environment and paths"""
    # 解决OpenMP问题 / Fix OpenMP issue
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 获取当前脚本的目录 / Get current script directory
    current_dir = Path(__file__).parent.absolute()
    
    # 添加当前目录到Python路径 / Add current directory to Python path
    sys.path.insert(0, str(current_dir))
    
    # 添加项目根目录到Python路径 / Add project root to Python path
    project_root = current_dir.parent.parent.parent.absolute()
    sys.path.insert(0, str(project_root))
    
    print(f"📁 当前目录: {current_dir}")
    print(f"📁 项目根目录: {project_root}")
    
    return project_root, current_dir

def main():
    """主函数 / Main function"""
    print("🚀 启动增强版3D面片可视化查询系统...")
    print("🚀 Starting Enhanced 3D Mesh Patch Visual Query System...")
    print("=" * 80)
    
    try:
        # 设置环境 / Setup environment
        project_root, current_dir = setup_environment()
        
        # 导入主模块 / Import main module
        print("📦 导入模块...")
        from main import main as run_main
        
        # 运行主程序 / Run main program
        print("▶️ 运行主程序...")
        run_main()
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("💡 请确保所有依赖模块都在正确的位置")
        print("💡 Please ensure all dependency modules are in the correct location")
        return 1
        
    except KeyboardInterrupt:
        print("\n👋 用户中断执行")
        print("👋 User interrupted execution")
        return 0
        
    except Exception as e:
        print(f"❌ 程序执行错误: {e}")
        print("❌ Program execution error")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)