#!/usr/bin/env python3
# File: run_project.py
# 项目启动脚本

import sys
from pathlib import Path

# 确保我们在正确的目录中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from src.common.path_manager import setup_project_environment
except ImportError:
    print("❌ 无法导入 'path_manager'。")
    print("请确保 'src/common/path_manager.py' 文件存在且无误。")
    # 提供一个备用方案来帮助诊断
    if not (project_root / 'src' / 'common' / 'path_manager.py').exists():
        print(f"错误: 文件不存在于 {(project_root / 'src' / 'common' / 'path_manager.py')}")
    sys.exit(1)


def main():
    """主启动函数"""
    print("🚀 启动项目环境设置")
    
    # 设置项目环境
    path_manager = setup_project_environment()
    path_manager.print_project_info()
    
    # 验证关键文件
    print("\n🔍 验证关键文件:")
    key_files = [
        ("数据库", path_manager.database_path),
        ("配置文件", path_manager.get_config_path()),
        ("模型目录", path_manager.model_dir),
    ]
    
    all_good = True
    for name, path in key_files:
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_good = False
    
    if all_good:
        print("\n🎉 项目环境设置完成，所有关键文件都存在！")
    else:
        print("\n⚠️  部分关键文件缺失，请检查项目结构")
    
    return path_manager

if __name__ == "__main__":
    main()
