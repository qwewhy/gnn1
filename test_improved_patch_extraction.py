#!/usr/bin/env python3
"""
测试改进后的patch提取系统
"""

import trimesh
import numpy as np
from pathlib import Path
import sys
import logging

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.train.data_processing.populate_db import ImprovedPatchExtractor, ImprovedGeometricFeatureExtractor

def test_single_model(model_path: Path):
    """测试单个模型的patch提取"""
    print(f"\n🔍 测试模型: {model_path.name}")
    
    try:
        # 加载模型
        mesh = trimesh.load(model_path, process=True)
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"❌ 不是有效的三角网格")
            return False
            
        print(f"   网格信息: {len(mesh.faces)} 面, {len(mesh.vertices)} 顶点")
        
        # 预处理
        mesh.merge_vertices()
        # 使用新的API替代过时的方法
        try:
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.update_faces(mesh.unique_faces())
        except AttributeError:
            # 如果新API不可用，使用旧方法（带警告抑制）
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mesh.remove_degenerate_faces()
                mesh.remove_duplicate_faces()
        
        # 使用改进的提取器
        extractor = ImprovedPatchExtractor()
        
        success_count = 0
        total_attempts = 10
        
        for i in range(total_attempts):
            patch_faces = extractor.extract_valid_patch(mesh, min_faces=5, max_faces=15)
            
            if patch_faces is not None:
                # 尝试提取几何特征
                boundary_info = extractor._compute_patch_boundary(mesh, patch_faces)
                
                if boundary_info is not None:
                    feature_extractor = ImprovedGeometricFeatureExtractor()
                    geometric_features = feature_extractor.extract_features_robust(
                        mesh, patch_faces, boundary_info
                    )
                    
                    if geometric_features is not None:
                        success_count += 1
                        print(f"   ✅ 成功提取patch {i+1}: {len(patch_faces)} 面, " +
                              f"{boundary_info['num_boundary_vertices']} 边界顶点")
        
        success_rate = success_count / total_attempts
        print(f"   📊 成功率: {success_rate:.1%} ({success_count}/{total_attempts})")
        
        return success_rate >= 0.5  # 至少50%成功率
        
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 测试改进后的patch提取系统")
    print("="*50)
    
    # 配置日志
    logging.basicConfig(level=logging.WARNING)  # 减少日志噪音
    
    # 找到模型文件
    model_dir = Path("model")
    if not model_dir.exists():
        print("❌ 模型目录不存在")
        return
    
    obj_files = list(model_dir.glob("**/*.obj"))
    if not obj_files:
        print("❌ 未找到模型文件")
        return
    
    print(f"找到 {len(obj_files)} 个模型文件")
    
    # 测试几个模型
    test_files = obj_files[:3]  # 只测试前3个
    
    results = []
    for obj_file in test_files:
        success = test_single_model(obj_file)
        results.append(success)
    
    # 汇总结果
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n📋 测试总结")
    print(f"测试模型数: {total_count}")
    print(f"成功模型数: {success_count}")
    print(f"总体成功率: {success_count/total_count:.1%}")
    
    if success_count >= total_count * 0.7:
        print("✅ 系统工作正常")
    else:
        print("⚠️ 系统可能需要进一步调整")

if __name__ == "__main__":
    main()
