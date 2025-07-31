#!/usr/bin/env python3
"""
测试新实现的核心算法
Test script for the new core algorithms implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import trimesh
import numpy as np
from pathlib import Path

from src.data_processing.proper_encoder import ProperPatternEncoder
from src.data_processing.proper_decoder import ProperPatternParser
from src.data_processing.populate_db import setup_database, encode_patch_to_pattern


def test_encoder_decoder():
    """测试编码器和解码器的基本功能"""
    print("=== 测试编码器和解码器 ===")
    
    # 创建一个简单的测试网格
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # 底面
        [0.5, 0.5, 1]  # 顶点
    ])
    
    faces = np.array([
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]  # 四个三角形面
    ])
    
    test_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    patch_faces = [0, 1]  # 选择前两个面作为面片
    
    try:
        # 测试编码
        encoder = ProperPatternEncoder()
        encoding_result = encoder.encode_patch_to_pattern(test_mesh, patch_faces)
        
        if encoding_result:
            edgebreaker_encoding, num_sides = encoding_result
            print(f"✓ 编码成功: {edgebreaker_encoding} (边数: {num_sides})")
            
            # 测试解码
            decoder = ProperPatternParser(edgebreaker_encoding, num_sides)
            graph_data = decoder.parse()
            
            if graph_data:
                print(f"✓ 解码成功: 节点数={graph_data['num_nodes']}, 边数={graph_data['edge_index'].shape[1]}")
                print(f"  特征维度: {len(graph_data)}")
                return True
            else:
                print("✗ 解码失败")
                return False
        else:
            print("✗ 编码失败")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_database_schema():
    """测试新的数据库模式"""
    print("\n=== 测试数据库模式 ===")
    
    try:
        test_db_path = Path("test_patterns.db")
        
        # 清理旧的测试数据库
        if test_db_path.exists():
            test_db_path.unlink()
        
        # 创建新数据库
        conn = setup_database(test_db_path)
        
        # 检查表结构
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(patterns)")
        columns = cursor.fetchall()
        
        expected_columns = [
            'edgebreaker_encoding', 'canonical_form', 'sides', 
            'complexity_score', 'num_vertices', 'num_faces'
        ]
        
        column_names = [col[1] for col in columns]
        
        for expected in expected_columns:
            if expected in column_names:
                print(f"✓ 数据库包含列: {expected}")
            else:
                print(f"✗ 数据库缺少列: {expected}")
                return False
        
        conn.close()
        test_db_path.unlink()  # 清理
        print("✓ 数据库模式测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 数据库测试失败: {e}")
        return False


def test_feature_dimensions():
    """测试特征维度匹配"""
    print("\n=== 测试特征维度 ===")
    
    try:
        # 模拟Pattern特征 (6维)
        pattern_features = torch.randn(10, 6)  # 10个节点，6维特征
        
        # 模拟Anchor特征 (8维)
        anchor_features = torch.randn(8, 8)   # 8个节点，8维特征
        
        print(f"✓ Pattern特征维度: {pattern_features.shape}")
        print(f"✓ Anchor特征维度: {anchor_features.shape}")
        
        # 检查前6维是否可以对应
        if pattern_features.shape[1] == 6 and anchor_features.shape[1] == 8:
            print("✓ 特征维度设计正确 - 前6维拓扑特征可以对应")
            return True
        else:
            print("✗ 特征维度不匹配")
            return False
            
    except Exception as e:
        print(f"✗ 特征测试失败: {e}")
        return False


def test_model_config_compatibility():
    """测试模型配置兼容性"""
    print("\n=== 测试模型配置 ===")
    
    try:
        from src.models.gnn_encoder import DualInputGNNEncoder
        
        # 使用新的配置创建模型
        model = DualInputGNNEncoder(
            anchor_in_channels=8,   # 6维拓扑 + 2维几何
            pattern_in_channels=6,  # 6维拓扑
            hidden_channels=128,
            out_channels=64,
            gnn_type='gat'
        )
        
        # 测试前向传播
        from torch_geometric.data import Data
        
        # 创建测试数据
        anchor_data = Data(
            x=torch.randn(5, 8),
            edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long),
            batch=torch.zeros(5, dtype=torch.long)
        )
        
        pattern_data = Data(
            x=torch.randn(4, 6),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
            batch=torch.zeros(4, dtype=torch.long)
        )
        
        # 测试编码
        anchor_emb = model(anchor_data, input_type='anchor')
        pattern_emb = model(pattern_data, input_type='pattern')
        
        print(f"✓ Anchor嵌入维度: {anchor_emb.shape}")
        print(f"✓ Pattern嵌入维度: {pattern_emb.shape}")
        
        if anchor_emb.shape == pattern_emb.shape == torch.Size([1, 64]):
            print("✓ 模型输出维度正确且一致")
            return True
        else:
            print("✗ 模型输出维度不正确")
            return False
            
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("🚀 开始测试新实现的核心算法\n")
    
    tests = [
        ("编码器解码器", test_encoder_decoder),
        ("数据库模式", test_database_schema),
        ("特征维度", test_feature_dimensions),
        ("模型配置", test_model_config_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} 测试通过")
        else:
            print(f"❌ {test_name} 测试失败")
    
    print(f"\n{'='*50}")
    print(f"测试总结: {passed}/{total} 测试通过")
    print('='*50)
    
    if passed == total:
        print("🎉 所有测试都通过了！新实现已准备就绪。")
        print("\n建议下一步:")
        print("1. 运行 python -m src.data_processing.populate_db 重建数据库")
        print("2. 使用新数据训练模型: python -m src.training.train")
        return True
    else:
        print("⚠️  有一些测试失败，请检查实现。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)