#!/usr/bin/env python3
"""
检查构建索引所需的文件和条件
Check build index requirements and conditions
"""

from pathlib import Path
import yaml
import sqlite3

def check_build_requirements():
    """检查构建索引的前提条件 Check prerequisites for building index"""
    print("🔍 检查索引构建要求 | Checking index build requirements")
    print("=" * 60)
    
    # 设置项目根目录 / Set project root directory
    project_root = Path(__file__).parent.absolute()
    print(f"📍 项目根目录: {project_root}")
    
    all_good = True
    
    # 1. 检查配置文件 Check config file
    config_path = project_root / 'configs' / 'config.yaml'
    if not config_path.exists():
        print("❌ 配置文件不存在 | Config file does not exist")
        all_good = False
        return all_good
    
    print("✅ 配置文件存在 | Config file exists")
    
    # 2. 加载配置 Load config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ 配置文件格式正确 | Config file format is correct")
    except Exception as e:
        print(f"❌ 配置文件格式错误: {e} | Config file format error: {e}")
        all_good = False
        return all_good
    
    # 3. 检查数据库文件 Check database file
    data_root = Path(config['data']['root'])
    db_path = data_root / 'raw' / 'patches.db'
    
    if not db_path.exists():
        print(f"❌ 数据库文件不存在: {db_path}")
        print("💡 需要先运行: python -m src.data_processing.populate_db")
        all_good = False
    else:
        db_size = db_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ 数据库文件存在 ({db_size:.1f} MB)")
        
        if db_size < 1:
            print("⚠️  数据库文件很小，可能是空的")
        
        # 检查数据库内容 Check database content
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM patterns")
            pattern_count = cursor.fetchone()[0]
            conn.close()
            print(f"📊 数据库包含 {pattern_count} 个模式 | Database contains {pattern_count} patterns")
            
            if pattern_count == 0:
                print("⚠️  数据库中没有模式数据")
                all_good = False
        except Exception as e:
            print(f"❌ 数据库查询失败: {e}")
            all_good = False
    
    # 4. 检查训练模型 Check trained models
    checkpoint_dir = project_root / config['training']['checkpoint_dir']
    model_files = {
        'best_model.pt': checkpoint_dir / 'best_model.pt',
        'final_model.pt': checkpoint_dir / 'final_model.pt',
        'latest_checkpoint.pt': checkpoint_dir / 'latest_checkpoint.pt'
    }
    
    found_model = False
    print(f"\n🤖 检查训练模型 | Checking trained models:")
    print(f"   检查点目录: {checkpoint_dir}")
    
    for name, path in model_files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {name} ({size_mb:.1f} MB)")
            found_model = True
        else:
            print(f"   ❌ {name} - 不存在")
    
    if not found_model:
        print("\n❌ 没有找到任何训练模型!")
        print("💡 您需要先训练模型:")
        print("   python -m src.training.improved_train --config configs/config.yaml")
        all_good = False
    
    # 5. 检查processed目录 Check processed directory
    processed_dir = data_root / 'processed'
    if not processed_dir.exists():
        print(f"\nℹ️  创建processed目录: {processed_dir}")
        processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ processed目录准备就绪: {processed_dir}")
    
    # 6. 检查Python模块 Check Python modules
    print(f"\n🐍 检查Python模块 | Checking Python modules:")
    required_modules = ['torch', 'torch_geometric', 'yaml', 'tqdm']
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} - 需要安装")
            all_good = False
    
    # 7. 检查build_index脚本 Check build_index script
    build_index_path = project_root / 'src' / 'inference' / 'build_index.py'
    if not build_index_path.exists():
        print(f"❌ build_index.py 不存在: {build_index_path}")
        all_good = False
    else:
        print(f"✅ build_index.py 存在")
    
    print(f"\n" + "=" * 60)
    if all_good:
        print("🎉 所有要求都已满足! | All requirements satisfied!")
        print("💡 现在可以运行索引构建命令:")
        print("   python -m src.inference.build_index --config configs/config.yaml")
    else:
        print("❌ 需要解决上述问题后才能构建索引")
        print("   Please resolve the above issues before building index")
    
    return all_good

if __name__ == "__main__":
    check_build_requirements()