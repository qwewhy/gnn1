#!/usr/bin/env python3
"""
简单测试脚本，验证triplet_generator的修复是否有效
Simple test script to verify triplet_generator fixes
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    print("Testing triplet generator fixes...")
    
    # 导入必要的模块
    from src.train.data_processing.pyg_dataset import PatchDataset
    from src.train.data_processing import TripletGenerator
    
    # 检查数据集是否存在
    data_root = project_root / "data"
    if not (data_root / "raw" / "patches.db").exists():
        print("Error: Database not found. Please run populate_db.py first.")
        sys.exit(1)
    
    # 加载数据集
    print("Loading dataset...")
    dataset = PatchDataset(root=str(data_root))
    print(f"Dataset loaded: {len(dataset)} patterns")
    
    # 测试三元组生成器
    print("Testing triplet generator...")
    
    # 使用第一个可用的网格文件
    model_dir = project_root / "model"
    obj_files = list(model_dir.glob("**/*.obj"))
    
    if not obj_files:
        print("Error: No .obj files found in model directory.")
        sys.exit(1)
        
    test_mesh = str(obj_files[0])
    print(f"Using test mesh: {Path(test_mesh).name}")
    
    # 创建生成器
    generator = TripletGenerator(test_mesh, dataset)
    
    # 生成一个三元组
    print("Generating triplet...")
    triplet = generator.generate_triplet()
    
    if triplet is None:
        print("Failed to generate triplet")
        sys.exit(1)
    
    anchor, positive, negative = triplet
    
    print(f"Success! Generated triplet:")
    print(f"  Anchor: {anchor.x.shape} features, {anchor.num_nodes} nodes")
    print(f"  Positive: {positive.x.shape} features, {positive.num_nodes} nodes") 
    print(f"  Negative: {negative.x.shape} features, {negative.num_nodes} nodes")
    
    print("\nAll tests passed! The dimension mismatch issue has been fixed.")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required packages are installed.")
    sys.exit(1)
except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Test completed successfully!")