#!/usr/bin/env python
# File: run_visual_query.py
# 项目根目录的可视化查询启动脚本 / Visual query startup script at project root

"""
从项目根目录启动可视化查询系统 / Start visual query system from project root

这个脚本位于项目根目录，可以直接运行，无需处理复杂的路径问题。
This script is located at the project root and can be run directly without dealing with complex path issues.

使用方法 / Usage:
    python run_visual_query.py
"""

import os
import sys
from pathlib import Path

def main():
    """主函数 / Main function"""
    print("🎨 增强版3D网格面片可视化查询系统")
    print("🎨 Enhanced 3D Mesh Patch Visual Query System")
    print("=" * 60)
    
    # 解决OpenMP问题 / Fix OpenMP issue
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 获取项目根目录 / Get project root directory
    project_root = Path(__file__).parent.absolute()
    
    # 添加项目根目录到Python路径 / Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    print(f"📁 项目根目录: {project_root}")
    
    try:
        # 导入可视化系统 / Import visualization system
        print("📦 导入可视化系统模块...")
        from src.use.visual_query_similar.main import main as visual_main
        
        # 运行可视化系统 / Run visualization system
        print("▶️ 启动可视化系统...")
        visual_main()
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("💡 请确保您在项目根目录运行此脚本")
        print("💡 Please ensure you run this script from the project root directory")
        
        # 尝试备用方案 / Try alternative approach
        print("\n🔄 尝试备用启动方案...")
        try:
            visual_dir = project_root / "src" / "use" / "visual_query_similar"
            sys.path.insert(0, str(visual_dir))
            from main import main as visual_main
            visual_main()
        except Exception as e2:
            print(f"❌ 备用方案也失败: {e2}")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 用户中断执行")
        return 0
        
    except Exception as e:
        print(f"❌ 程序执行错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("✅ 程序执行完成")
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)