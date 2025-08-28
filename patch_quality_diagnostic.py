#!/usr/bin/env python3
"""
Patch质量诊断工具
用于检查已提取的patch数据的质量和有效性
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import logging
import argparse
import sys

class PatchQualityDiagnostic:
    """Patch质量诊断器"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    def run_full_diagnostic(self) -> dict:
        """运行完整的诊断检查"""
        print("🔍 开始Patch质量诊断...")
        
        results = {
            'basic_stats': self.check_basic_statistics(),
            'geometric_validity': self.check_geometric_validity(),
            'encoding_quality': self.check_encoding_quality(),
            'boundary_analysis': self.check_boundary_analysis(),
            'data_completeness': self.check_data_completeness()
        }
        
        # 生成诊断报告
        self.generate_diagnostic_report(results)
        
        return results
    
    def check_basic_statistics(self) -> dict:
        """检查基本统计信息"""
        print("📊 检查基本统计...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 总数统计
        cursor.execute("SELECT COUNT(*) FROM patterns")
        total_patterns = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM patterns WHERE quality='new'")
        new_patterns = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM patterns WHERE quality='old'")
        old_patterns = cursor.fetchone()[0]
        
        # 边数分布
        cursor.execute("SELECT sides, COUNT(*) FROM patterns GROUP BY sides ORDER BY sides")
        sides_distribution = dict(cursor.fetchall())
        
        # 复杂度分布
        cursor.execute("SELECT complexity_score FROM patterns WHERE complexity_score IS NOT NULL")
        complexity_scores = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        stats = {
            'total_patterns': total_patterns,
            'new_patterns': new_patterns,
            'old_patterns': old_patterns,
            'sides_distribution': sides_distribution,
            'complexity_stats': {
                'mean': np.mean(complexity_scores) if complexity_scores else 0,
                'std': np.std(complexity_scores) if complexity_scores else 0,
                'min': np.min(complexity_scores) if complexity_scores else 0,
                'max': np.max(complexity_scores) if complexity_scores else 0
            }
        }
        
        print(f"   总Pattern数: {total_patterns}")
        print(f"   'new'类型: {new_patterns}")
        print(f"   'old'类型: {old_patterns}")
        print(f"   边数分布: {sides_distribution}")
        
        return stats
    
    def check_geometric_validity(self) -> dict:
        """检查几何特征的有效性"""
        print("🎯 检查几何特征有效性...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, boundary_vertices, mean_curvatures, gaussian_curvatures, 
                   edge_lengths, area, total_boundary_length
            FROM patterns 
            WHERE boundary_vertices IS NOT NULL
            """)
        
        patterns_with_geometry = cursor.fetchall()
        conn.close()
        
        validity_issues = {
            'invalid_boundary_vertices': 0,
            'nan_curvatures': 0,
            'infinite_curvatures': 0,
            'zero_area': 0,
            'negative_edge_lengths': 0,
            'mismatched_dimensions': 0,
            'total_checked': len(patterns_with_geometry)
        }
        
        problematic_patterns = []
        
        for pattern_data in patterns_with_geometry:
            pattern_id, boundary_vertices_json, mean_curv_json, gauss_curv_json, \
            edge_lengths_json, area, total_boundary_length = pattern_data
            
            issues = []
            
            try:
                # 检查边界顶点
                boundary_vertices = json.loads(boundary_vertices_json)
                if not boundary_vertices or len(boundary_vertices) < 3:
                    validity_issues['invalid_boundary_vertices'] += 1
                    issues.append("无效边界顶点数")
                
                # 检查维度一致性
                if mean_curv_json and gauss_curv_json and edge_lengths_json:
                    mean_curv = json.loads(mean_curv_json)
                    gauss_curv = json.loads(gauss_curv_json)
                    edge_lengths = json.loads(edge_lengths_json)
                    
                    if not (len(mean_curv) == len(gauss_curv) == len(edge_lengths) == len(boundary_vertices)):
                        validity_issues['mismatched_dimensions'] += 1
                        issues.append("特征维度不匹配")
                    
                    # 检查曲率数值
                    if any(np.isnan(mean_curv)) or any(np.isnan(gauss_curv)):
                        validity_issues['nan_curvatures'] += 1
                        issues.append("曲率包含NaN")
                    
                    if any(np.isinf(mean_curv)) or any(np.isinf(gauss_curv)):
                        validity_issues['infinite_curvatures'] += 1
                        issues.append("曲率包含无穷大")
                    
                    # 检查边长
                    if any(length <= 0 for length in edge_lengths):
                        validity_issues['negative_edge_lengths'] += 1
                        issues.append("存在非正边长")
                
                # 检查面积
                if area is not None and area <= 0:
                    validity_issues['zero_area'] += 1
                    issues.append("面积为零或负数")
                
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                issues.append(f"数据解析错误: {e}")
            
            if issues:
                problematic_patterns.append({
                    'pattern_id': pattern_id,
                    'issues': issues
                })
        
        validity_ratio = 1 - len(problematic_patterns) / max(len(patterns_with_geometry), 1)
        
        print(f"   检查了 {len(patterns_with_geometry)} 个有几何特征的pattern")
        print(f"   几何有效性比率: {validity_ratio:.2%}")
        print(f"   主要问题: NaN曲率({validity_issues['nan_curvatures']})," +
              f"维度不匹配({validity_issues['mismatched_dimensions']})")
        
        return {
            'validity_issues': validity_issues,
            'problematic_patterns': problematic_patterns[:10],  # 只返回前10个问题样本
            'validity_ratio': validity_ratio
        }
    
    def check_encoding_quality(self) -> dict:
        """检查编码质量"""
        print("🔐 检查编码质量...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT edgebreaker_encoding, canonical_form FROM patterns")
        encodings = cursor.fetchall()
        conn.close()
        
        encoding_stats = {
            'empty_encodings': 0,
            'very_short_encodings': 0,  # 长度 < 5
            'very_long_encodings': 0,   # 长度 > 100
            'invalid_characters': 0,
            'encoding_length_distribution': defaultdict(int),
            'character_frequency': Counter(),
            'total_encodings': len(encodings)
        }
        
        valid_edgebreaker_chars = set('SCLRE')
        
        for edgebreaker_enc, canonical_form in encodings:
            enc_len = len(edgebreaker_enc) if edgebreaker_enc else 0
            
            if enc_len == 0:
                encoding_stats['empty_encodings'] += 1
            elif enc_len < 5:
                encoding_stats['very_short_encodings'] += 1
            elif enc_len > 100:
                encoding_stats['very_long_encodings'] += 1
            
            # 长度分布（分组）
            length_group = (enc_len // 10) * 10  # 0-9, 10-19, 20-29, ...
            encoding_stats['encoding_length_distribution'][length_group] += 1
            
            # 字符频率
            if edgebreaker_enc:
                for char in edgebreaker_enc:
                    encoding_stats['character_frequency'][char] += 1
                    if char not in valid_edgebreaker_chars:
                        encoding_stats['invalid_characters'] += 1
        
        quality_ratio = 1 - (encoding_stats['empty_encodings'] + 
                           encoding_stats['invalid_characters']) / max(len(encodings), 1)
        
        print(f"   总编码数: {len(encodings)}")
        print(f"   编码质量比率: {quality_ratio:.2%}")
        print(f"   空编码: {encoding_stats['empty_encodings']}")
        print(f"   异常字符: {encoding_stats['invalid_characters']}")
        
        return {
            'encoding_stats': dict(encoding_stats),
            'quality_ratio': quality_ratio
        }
    
    def check_boundary_analysis(self) -> dict:
        """分析边界特性"""
        print("🔲 分析边界特性...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT sides, total_boundary_length, area FROM patterns WHERE area > 0")
        boundary_data = cursor.fetchall()
        conn.close()
        
        boundary_stats = {
            'aspect_ratios': [],  # 边界长度/面积比
            'side_counts': Counter(),
            'unusual_boundaries': 0  # 异常边界（比如sides < 3或过大）
        }
        
        for sides, boundary_length, area in boundary_data:
            boundary_stats['side_counts'][sides] += 1
            
            if sides < 3 or sides > 50:  # 异常边界边数
                boundary_stats['unusual_boundaries'] += 1
            
            if boundary_length > 0 and area > 0:
                aspect_ratio = boundary_length / np.sqrt(area)  # 标准化的形状比
                boundary_stats['aspect_ratios'].append(aspect_ratio)
        
        # 计算形状比统计
        if boundary_stats['aspect_ratios']:
            aspect_ratios = boundary_stats['aspect_ratios']
            boundary_stats['aspect_ratio_stats'] = {
                'mean': np.mean(aspect_ratios),
                'std': np.std(aspect_ratios),
                'median': np.median(aspect_ratios),
                'outliers': len([r for r in aspect_ratios if r > np.mean(aspect_ratios) + 2*np.std(aspect_ratios)])
            }
        
        print(f"   边界分析样本数: {len(boundary_data)}")
        print(f"   异常边界数: {boundary_stats['unusual_boundaries']}")
        print(f"   最常见边数: {boundary_stats['side_counts'].most_common(3)}")
        
        return boundary_stats
    
    def check_data_completeness(self) -> dict:
        """检查数据完整性"""
        print("✅ 检查数据完整性...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查各字段的NULL值比例
        fields_to_check = [
            'edgebreaker_encoding', 'canonical_form', 'sides', 'complexity_score',
            'num_vertices', 'num_faces', 'source_obj', 'quality',
            'boundary_vertices', 'mean_curvatures', 'area'
        ]
        
        completeness_stats = {}
        
        for field in fields_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM patterns WHERE {field} IS NOT NULL")
            non_null_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM patterns")
            total_count = cursor.fetchone()[0]
            
            completeness_ratio = non_null_count / max(total_count, 1)
            completeness_stats[field] = {
                'non_null_count': non_null_count,
                'total_count': total_count,
                'completeness_ratio': completeness_ratio
            }
        
        conn.close()
        
        # 找出完整性最低的字段
        lowest_completeness = min(
            completeness_stats.values(), 
            key=lambda x: x['completeness_ratio']
        )
        
        print(f"   数据完整性检查完成")
        print(f"   最低完整性字段: {lowest_completeness['completeness_ratio']:.2%}")
        
        return completeness_stats
    
    def generate_diagnostic_report(self, results: dict):
        """生成诊断报告"""
        print("\n" + "="*60)
        print("📋 PATCH质量诊断报告")
        print("="*60)
        
        # 总体评分
        geometric_score = results['geometric_validity']['validity_ratio']
        encoding_score = results['encoding_quality']['quality_ratio'] 
        
        overall_score = (geometric_score + encoding_score) / 2
        
        print(f"\n🎯 总体质量评分: {overall_score:.2%}")
        
        if overall_score >= 0.8:
            quality_level = "✅ 优秀"
        elif overall_score >= 0.6:
            quality_level = "⚠️ 良好（需要改进）"
        else:
            quality_level = "❌ 较差（需要重新提取）"
        
        print(f"📊 质量等级: {quality_level}")
        
        # 主要问题汇总
        print(f"\n🚨 主要问题:")
        
        geom_issues = results['geometric_validity']['validity_issues']
        if geom_issues['nan_curvatures'] > 0:
            print(f"   - {geom_issues['nan_curvatures']} 个pattern有NaN曲率值")
        
        if geom_issues['mismatched_dimensions'] > 0:
            print(f"   - {geom_issues['mismatched_dimensions']} 个pattern特征维度不匹配")
        
        enc_issues = results['encoding_quality']['encoding_stats']
        if enc_issues['empty_encodings'] > 0:
            print(f"   - {enc_issues['empty_encodings']} 个pattern编码为空")
        
        # 建议
        print(f"\n💡 改进建议:")
        
        if geometric_score < 0.7:
            print("   - 几何特征提取需要改进，建议使用更鲁棒的方法")
        
        if encoding_score < 0.7:
            print("   - EdgeBreaker编码质量较差，建议检查编码算法")
        
        if results['boundary_analysis']['unusual_boundaries'] > 0:
            print("   - 存在异常边界，建议增加边界有效性检查")
        
        print("="*60)


def main():
    """主函数 - 运行诊断"""
    parser = argparse.ArgumentParser(description="Patch质量诊断工具")
    parser.add_argument("--db-path", required=True, help="数据库文件路径")
    parser.add_argument("--output-dir", help="诊断结果输出目录")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 运行诊断
        diagnostic = PatchQualityDiagnostic(args.db_path)
        results = diagnostic.run_full_diagnostic()
        
        # 如果指定了输出目录，保存详细结果
        if args.output_dir:
            output_path = Path(args.output_dir) / "patch_diagnostic_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            
            print(f"\n📁 详细诊断结果已保存至: {output_path}")
            
    except Exception as e:
        print(f"❌ 诊断过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
