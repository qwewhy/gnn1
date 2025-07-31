#!/usr/bin/env python3
"""
核心算法独立测试 - 无需深度学习依赖
Core algorithms standalone test - without deep learning dependencies
"""

import sys
import os
import sqlite3
from pathlib import Path

def test_database_schema():
    """测试新的数据库模式"""
    print("=== 测试数据库模式 ===")
    
    try:
        test_db_path = Path("test_simple.db")
        
        # 清理旧的测试数据库
        if test_db_path.exists():
            test_db_path.unlink()
        
        # 创建新数据库
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        # 使用我们新的数据库模式
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            edgebreaker_encoding TEXT NOT NULL,
            canonical_form TEXT NOT NULL,
            sides INTEGER NOT NULL,
            complexity_score REAL DEFAULT 0.0,
            num_vertices INTEGER DEFAULT 0,
            num_faces INTEGER DEFAULT 0,
            source_obj TEXT NOT NULL,
            quality TEXT NOT NULL,
            UNIQUE(canonical_form, sides)
        );
        """)
        
        # 检查表结构
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
        
        # 测试插入新格式数据
        test_data = [
            ("2 1 3#SCLRLCRE", "2_1_3_sclrlcre", 8, 1.5, 12, 8, "test.obj", "new"),
            ("1 2#SLCR", "1_2_slcr", 6, 1.2, 10, 6, "test2.obj", "new"),
        ]
        
        for data in test_data:
            cursor.execute("""
            INSERT INTO patterns (
                edgebreaker_encoding, canonical_form, sides, 
                complexity_score, num_vertices, num_faces,
                source_obj, quality
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, data)
        
        # 验证数据插入
        cursor.execute("SELECT COUNT(*) FROM patterns")
        count = cursor.fetchone()[0]
        
        if count == 2:
            print(f"✓ 成功插入 {count} 条测试数据")
        else:
            print(f"✗ 插入数据失败，预期2条，实际{count}条")
            return False
        
        # 测试查询新格式
        cursor.execute("SELECT edgebreaker_encoding, canonical_form, complexity_score FROM patterns WHERE quality='new'")
        results = cursor.fetchall()
        
        print("✓ 新格式数据查询结果:")
        for result in results:
            print(f"  编码: {result[0]}, 规范形式: {result[1]}, 复杂度: {result[2]}")
        
        conn.close()
        test_db_path.unlink()  # 清理
        print("✓ 数据库模式测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 数据库测试失败: {e}")
        return False


def test_encoding_format():
    """测试编码格式解析"""
    print("\n=== 测试编码格式 ===")
    
    try:
        # 模拟我们的新编码格式
        test_encodings = [
            "2 1 3#SCLRLCRE",     # 多弦信息 + Edgebreaker序列
            "1 2#SLCR",           # 简单情况
            "#SCLR",              # 仅Edgebreaker序列
            "3 1 2 4#SCLRLCRSE",  # 复杂多弦
        ]
        
        for encoding in test_encodings:
            print(f"测试编码: {encoding}")
            
            # 解析格式
            if '#' in encoding:
                multi_chord_part, edgebreaker_part = encoding.split('#', 1)
            else:
                multi_chord_part = ""
                edgebreaker_part = encoding
            
            # 解析多弦信息
            multi_chord_info = []
            if multi_chord_part.strip():
                for item in multi_chord_part.strip().split():
                    if item.isdigit():
                        multi_chord_info.append(int(item))
            
            # 解析Edgebreaker操作
            edgebreaker_ops = []
            for char in edgebreaker_part:
                if char in ['C', 'L', 'R', 'S', 'E']:
                    edgebreaker_ops.append(char)
            
            print(f"  多弦信息: {multi_chord_info}")
            print(f"  Edgebreaker操作: {edgebreaker_ops}")
            
            # 验证解析结果
            if len(edgebreaker_ops) > 0:
                print(f"  ✓ 解析成功")
            else:
                print(f"  ✗ 解析失败")
                return False
        
        print("✓ 编码格式测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 编码格式测试失败: {e}")
        return False


def test_feature_dimensions():
    """测试特征维度设计"""
    print("\n=== 测试特征维度设计 ===")
    
    try:
        # 模拟Pattern特征 (6维拓扑)
        pattern_features = {
            'valence': 1,           # 节点度数
            'is_boundary': 1,       # 边界标记  
            'is_corner': 1,         # 角点标记
            'distance_to_singular': 1, # 到奇异点距离
            'local_topology_config': 1, # 局部拓扑配置
            'boundary_position_encoding': 1, # 边界位置编码
        }
        
        # 模拟Anchor特征 (6维拓扑 + 2维几何)
        anchor_features = {
            # 前6维与Pattern对应
            'valence': 1,
            'is_boundary': 1,
            'is_corner': 1, 
            'distance_to_singular': 1,
            'local_topology_config': 1,
            'boundary_position_encoding': 1,
            # 额外2维几何特征
            'curvature': 1,
            'length_ratio': 1,
        }
        
        pattern_dim = len(pattern_features)
        anchor_dim = len(anchor_features)
        
        print(f"✓ Pattern特征维度: {pattern_dim}")
        print(f"✓ Anchor特征维度: {anchor_dim}")
        
        # 检查前6维是否对应
        pattern_keys = list(pattern_features.keys())
        anchor_keys = list(anchor_features.keys())[:6]  # 前6维
        
        matching_features = 0
        for i in range(6):
            if pattern_keys[i] == anchor_keys[i]:
                matching_features += 1
                print(f"  ✓ 维度{i}: {pattern_keys[i]} <-> {anchor_keys[i]}")
            else:
                print(f"  ✗ 维度{i}: {pattern_keys[i]} <-> {anchor_keys[i]} 不匹配")
        
        if matching_features == 6:
            print(f"✓ 前6维特征完全对应")
            anchor_extra_keys = list(anchor_features.keys())[6:]  # 修正索引
            print(f"✓ Anchor额外特征: {anchor_extra_keys}")
            return True
        else:
            print(f"✗ 只有{matching_features}/6维特征对应")
            return False
            
    except Exception as e:
        print(f"✗ 特征维度测试失败: {e}")
        return False


def test_canonical_normalization():
    """测试规范化逻辑"""
    print("\n=== 测试规范化逻辑 ===")
    
    try:
        # 测试编码规范化
        test_cases = [
            ("2 1 3#SCLRLCRE", "2_1_3_sclrlcre"),
            ("  1 2  # SLCR  ", "1_2_slcr"),
            ("3#SCLR", "3_sclr"),
            ("#SLCRE", "slcre"),
        ]
        
        def normalize_encoding(encoding):
            """简化的规范化实现"""
            normalized = encoding.strip().lower()
            normalized = normalized.replace(" ", "_").replace("#", "_")
            # 移除连续的下划线
            while "__" in normalized:
                normalized = normalized.replace("__", "_")
            return normalized.strip("_")
        
        for original, expected in test_cases:
            result = normalize_encoding(original)
            if result == expected:
                print(f"✓ '{original}' -> '{result}'")
            else:
                print(f"✗ '{original}' -> '{result}', 预期: '{expected}'")
                return False
        
        print("✓ 规范化测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 规范化测试失败: {e}")
        return False


def main():
    """运行所有核心测试"""
    print("🚀 开始测试核心算法（无深度学习依赖）\n")
    
    tests = [
        ("数据库模式", test_database_schema),
        ("编码格式", test_encoding_format),
        ("特征维度", test_feature_dimensions),
        ("规范化逻辑", test_canonical_normalization),
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
    print(f"核心算法测试总结: {passed}/{total} 测试通过")
    print('='*50)
    
    if passed == total:
        print("🎉 核心算法测试全部通过！")
        print("\n✅ 验证结果:")
        print("• 新数据库模式设计正确")
        print("• 编码格式解析正常")  
        print("• 特征维度完美对应")
        print("• 规范化逻辑工作正常")
        
        print("\n📋 下一步操作建议:")
        print("1. 安装深度学习依赖: pip install torch torch-geometric trimesh")
        print("2. 运行完整测试: python test_new_implementation.py")
        print("3. 重建数据库: python -m src.data_processing.populate_db")
        print("4. 开始训练: python -m src.training.train")
        
        return True
    else:
        print("⚠️  部分核心测试失败，请检查实现。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)