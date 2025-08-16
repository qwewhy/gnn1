# File: src/train/data_processing/triplet_mining_example.py
# 硬三元组挖掘使用示例 / Hard triplet mining usage example

import torch
import numpy as np
from typing import Dict, List
import logging
from pathlib import Path

from src.train.data_processing.triplet_generator import TripletGenerator
from src.train.data_processing.pyg_dataset import PatchDataset
from src.train.models.improved_gat_encoder import ImprovedGATEncoder

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_hard_mining_config() -> Dict:
    """
    创建硬挖掘配置
    """
    return {
        'enabled': True,
        'mining_type': 'adaptive',  # 'hard', 'semi_hard', 'adaptive'
        'margin': 0.5,
        'hard_ratio': 0.4,
        'semi_hard_ratio': 0.4,
        'easy_ratio': 0.2
    }

def demonstrate_geometric_vs_hard_mining():
    """
    演示几何挖掘与硬挖掘的对比
    """
    # 模拟数据路径（需要根据实际情况调整）
    mesh_path = "model/stanford-bunny-retopo.obj"
    dataset_path = "data/processed/pyg_patch_dataset_with_geometry.pt"
    
    try:
        # 1. 加载数据集
        patch_dataset = PatchDataset(root="data/processed", name="pyg_patch_dataset_with_geometry")
        logger.info(f"加载数据集成功，包含 {len(patch_dataset)} 个样本")
        
        # 2. 创建几何三元组生成器（不使用硬挖掘）
        geometric_generator = TripletGenerator(
            mesh_path=mesh_path,
            patch_dataset=patch_dataset,
            hard_mining_config={'enabled': False}
        )
        
        # 3. 创建硬挖掘三元组生成器
        hard_mining_config = create_hard_mining_config()
        hard_generator = TripletGenerator(
            mesh_path=mesh_path,
            patch_dataset=patch_dataset,
            hard_mining_config=hard_mining_config
        )
        
        # 4. 创建简单的编码器模型（用于硬挖掘）
        if len(patch_dataset) > 0:
            sample_data = patch_dataset[0]
            input_dim = sample_data.x.size(1) if hasattr(sample_data, 'x') else 8
            encoder_model = ImprovedGATEncoder(
                input_dim=input_dim,
                hidden_dim=64,
                output_dim=128,
                num_heads=4,
                num_layers=3
            )
            encoder_model.eval()
        else:
            logger.error("数据集为空，无法创建编码器")
            return
        
        # 5. 测试几何三元组生成
        logger.info("\n=== 几何三元组生成测试 ===")
        geometric_triplets = geometric_generator.generate_batch_triplets(batch_size=8)
        
        if geometric_triplets is not None:
            anchors, positives, negatives = geometric_triplets
            logger.info(f"几何方法生成了 {len(anchors)} 个三元组")
            
            # 分析三元组质量
            analyze_triplet_quality(anchors, positives, negatives, "几何方法")
        else:
            logger.warning("几何方法未能生成三元组")
        
        # 6. 测试硬挖掘三元组生成
        logger.info("\n=== 硬挖掘三元组生成测试 ===")
        hard_triplets = hard_generator.generate_batch_triplets(
            batch_size=8,
            encoder_model=encoder_model
        )
        
        if hard_triplets is not None:
            anchors, positives, negatives = hard_triplets
            logger.info(f"硬挖掘方法生成了 {len(anchors)} 个三元组")
            
            # 分析三元组质量
            analyze_triplet_quality(anchors, positives, negatives, "硬挖掘方法")
            
            # 显示挖掘统计信息
            stats = hard_generator.get_hard_mining_statistics()
            if stats:
                logger.info(f"硬挖掘统计: {stats}")
        else:
            logger.warning("硬挖掘方法未能生成三元组")
            
    except Exception as e:
        logger.error(f"演示过程出错: {e}")
        logger.info("这可能是由于缺少模型文件或数据集，这是正常的演示错误")

def analyze_triplet_quality(anchors: List, positives: List, negatives: List, method_name: str):
    """
    分析三元组质量
    """
    logger.info(f"\n{method_name} 三元组质量分析:")
    logger.info(f"  锚点数量: {len(anchors)}")
    logger.info(f"  正样本数量: {len(positives)}")
    logger.info(f"  负样本数量: {len(negatives)}")
    
    if len(anchors) > 0:
        # 分析节点数分布
        anchor_node_counts = [data.num_nodes for data in anchors]
        positive_node_counts = [data.num_nodes for data in positives]
        negative_node_counts = [data.num_nodes for data in negatives]
        
        logger.info(f"  锚点平均节点数: {np.mean(anchor_node_counts):.2f}")
        logger.info(f"  正样本平均节点数: {np.mean(positive_node_counts):.2f}")
        logger.info(f"  负样本平均节点数: {np.mean(negative_node_counts):.2f}")
        
        # 分析拓扑匹配情况
        topology_matches = sum(1 for a, p in zip(anchors, positives) 
                             if a.num_sides == p.num_sides)
        topology_mismatches = sum(1 for a, n in zip(anchors, negatives) 
                                if a.num_sides == n.num_sides)
        
        logger.info(f"  锚点-正样本拓扑匹配率: {topology_matches/len(anchors)*100:.1f}%")
        logger.info(f"  锚点-负样本拓扑匹配率: {topology_mismatches/len(anchors)*100:.1f}%")

def benchmark_mining_performance():
    """
    性能基准测试
    """
    logger.info("\n=== 挖掘性能基准测试 ===")
    
    # 不同配置的测试
    configs = [
        {'mining_type': 'hard', 'margin': 0.5},
        {'mining_type': 'semi_hard', 'margin': 0.5},
        {'mining_type': 'adaptive', 'margin': 0.5, 'hard_ratio': 0.3, 'semi_hard_ratio': 0.5, 'easy_ratio': 0.2}
    ]
    
    for config in configs:
        config['enabled'] = True
        logger.info(f"\n测试配置: {config}")
        
        # 这里可以添加具体的性能测试代码
        # 由于需要真实数据，这里只是展示框架
        logger.info(f"  配置类型: {config['mining_type']}")
        logger.info(f"  边界值: {config['margin']}")

def create_training_integration_example():
    """
    展示如何在训练中集成硬挖掘
    """
    logger.info("\n=== 训练集成示例 ===")
    
    code_example = '''
# 在训练循环中使用硬挖掘
def train_with_hard_mining(model, triplet_generator, train_loader, optimizer):
    model.train()
    
    for batch_idx, batch_data in enumerate(train_loader):
        # 获取硬挖掘三元组
        triplets = triplet_generator.generate_batch_triplets(
            batch_size=len(batch_data),
            encoder_model=model
        )
        
        if triplets is None:
            continue
            
        anchors, positives, negatives = triplets
        
        # 计算嵌入
        anchor_embeddings = model(anchors)
        positive_embeddings = model(positives)
        negative_embeddings = model(negatives)
        
        # 计算三元组损失
        triplet_loss = F.triplet_margin_loss(
            anchor_embeddings, 
            positive_embeddings, 
            negative_embeddings,
            margin=0.5
        )
        
        # 反向传播
        optimizer.zero_grad()
        triplet_loss.backward()
        optimizer.step()
        
        # 记录统计信息
        if batch_idx % 100 == 0:
            stats = triplet_generator.get_hard_mining_statistics()
            print(f"Batch {batch_idx}, Loss: {triplet_loss:.4f}, Mining Stats: {stats}")
    '''
    
    logger.info("训练集成代码示例:")
    logger.info(code_example)

if __name__ == '__main__':
    logger.info("硬三元组挖掘系统演示")
    
    # 运行演示
    demonstrate_geometric_vs_hard_mining()
    benchmark_mining_performance()
    create_training_integration_example()
    
    logger.info("\n演示完成！")
    logger.info("注意：如果出现文件未找到错误，这是正常的，因为演示需要实际的模型文件和数据集。")