# File: src/train/training/hard_triplet_mining.py
# 硬三元组挖掘系统 / Hard triplet mining system

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

@dataclass
class TripletBatch:
    """三元组批次数据"""
    anchors: torch.Tensor      # 锚点嵌入
    positives: torch.Tensor    # 正样本嵌入
    negatives: torch.Tensor    # 负样本嵌入
    anchor_labels: torch.Tensor  # 锚点标签
    positive_labels: torch.Tensor  # 正样本标签
    negative_labels: torch.Tensor  # 负样本标签
    mining_info: Dict         # 挖掘信息

class TripletMiner(ABC):
    """三元组挖掘器抽象基类"""
    
    @abstractmethod
    def mine_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """挖掘三元组"""
        pass

class BatchHardMiner(TripletMiner):
    """批次硬三元组挖掘器"""
    
    def __init__(self, margin: float = 0.3, squared: bool = False, relaxed_threshold: float = 0.8):
        """
        初始化批次硬挖掘器
        
        Args:
            margin: 三元组损失边界
            squared: 是否使用平方欧氏距离
            relaxed_threshold: 宽松阈值比例（用于更容易找到硬三元组）
        """
        self.margin = margin
        self.squared = squared
        self.relaxed_threshold = relaxed_threshold
        self.min_triplets = 1  # 确保至少生成一些三元组
        self.logger = logging.getLogger(__name__)
    
    def mine_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从批次中挖掘硬三元组
        
        Args:
            embeddings: 嵌入向量 [batch_size, embedding_dim]
            labels: 标签 [batch_size]
            
        Returns:
            (anchor_indices, positive_indices, negative_indices)
        """
        batch_size = embeddings.size(0)
        
        # 计算所有样本间的距离矩阵
        distance_matrix = self._compute_distance_matrix(embeddings)
        
        # 创建标签匹配矩阵
        labels = labels.view(-1, 1)
        label_equal = labels == labels.t()  # [batch_size, batch_size]
        
        # 为每个锚点找到最难的正样本和负样本
        anchor_indices = []
        positive_indices = []
        negative_indices = []
        
        for i in range(batch_size):
            # 找到正样本（相同标签但不是自己）
            positive_mask = label_equal[i] & (torch.arange(batch_size, device=embeddings.device) != i)
            if not positive_mask.any():
                continue
                
            # 找到负样本（不同标签）
            negative_mask = ~label_equal[i]
            if not negative_mask.any():
                continue
            
            # 选择最难的正样本（距离最远的正样本）
            positive_distances = distance_matrix[i][positive_mask]
            hardest_positive_idx = positive_mask.nonzero(as_tuple=True)[0][positive_distances.argmax()]
            
            # 选择最难的负样本（距离最近的负样本）
            negative_distances = distance_matrix[i][negative_mask]
            hardest_negative_idx = negative_mask.nonzero(as_tuple=True)[0][negative_distances.argmin()]
            
            # 检查是否构成有效的硬三元组
            pos_dist = distance_matrix[i, hardest_positive_idx]
            neg_dist = distance_matrix[i, hardest_negative_idx]
            
            # 使用更宽松的条件：降低margin阈值
            relaxed_margin = self.margin * self.relaxed_threshold
            if pos_dist + relaxed_margin > neg_dist:  # 违反边界的硬三元组
                anchor_indices.append(i)
                positive_indices.append(hardest_positive_idx)
                negative_indices.append(hardest_negative_idx)
        
        if not anchor_indices:
            # 如果没有硬三元组，尝试半硬挖掘作为回退
            self.logger.warning(f"未找到硬三元组（margin={self.margin:.3f}），尝试半硬挖掘")
            return self._fallback_semi_hard_mining(embeddings, labels, distance_matrix)
        
        self.logger.debug(f"成功挖掘到 {len(anchor_indices)} 个硬三元组")
        
        return (torch.tensor(anchor_indices, device=embeddings.device),
                torch.tensor(positive_indices, device=embeddings.device),
                torch.tensor(negative_indices, device=embeddings.device))
    
    def _compute_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """计算距离矩阵"""
        if self.squared:
            # 平方欧氏距离
            distances = torch.cdist(embeddings, embeddings, p=2) ** 2
        else:
            # 欧氏距离
            distances = torch.cdist(embeddings, embeddings, p=2)
        
        return distances
    
    def _fallback_semi_hard_mining(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                                   distance_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """回退到半硬挖掘"""
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        labels = labels.view(-1, 1)
        label_equal = labels == labels.t()
        
        anchor_indices = []
        positive_indices = []
        negative_indices = []
        
        for i in range(batch_size):
            positive_mask = label_equal[i] & (torch.arange(batch_size, device=device) != i)
            negative_mask = ~label_equal[i]
            
            if not positive_mask.any() or not negative_mask.any():
                continue
            
            # 选择任意正样本
            pos_idx = positive_mask.nonzero(as_tuple=True)[0][0]
            pos_dist = distance_matrix[i, pos_idx]
            
            # 找到满足半硬条件的负样本
            neg_distances = distance_matrix[i][negative_mask]
            neg_indices = negative_mask.nonzero(as_tuple=True)[0]
            
            # 半硬条件：pos_dist < neg_dist < pos_dist + margin
            semi_hard_mask = (neg_distances > pos_dist) & (neg_distances < pos_dist + self.margin)
            
            if semi_hard_mask.any():
                valid_neg_indices = neg_indices[semi_hard_mask]
                chosen_neg_idx = valid_neg_indices[0]  # 选择第一个满足条件的
                
                anchor_indices.append(i)
                positive_indices.append(pos_idx)
                negative_indices.append(chosen_neg_idx)
        
        if anchor_indices:
            self.logger.info(f"半硬挖掘成功：找到 {len(anchor_indices)} 个半硬三元组")
            return (torch.tensor(anchor_indices, device=device),
                    torch.tensor(positive_indices, device=device),
                    torch.tensor(negative_indices, device=device))
        else:
            # 最终回退到智能随机采样
            return self._fallback_smart_random_sampling(batch_size, labels, device)
    
    def _fallback_smart_random_sampling(self, batch_size: int, labels: torch.Tensor, 
                                       device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """智能随机采样（确保标签正确性）"""
        self.logger.warning("所有挖掘策略失败，使用智能随机采样")
        
        # 创建标签映射
        unique_labels = torch.unique(labels)
        if len(unique_labels) < 2:
            # 如果标签类别不足，使用简单随机采样
            count = max(1, batch_size // 6)
            indices = torch.randperm(batch_size, device=device)
            return indices[:count], indices[count:2*count], indices[2*count:3*count]
        
        label_to_indices = {}
        for label in unique_labels:
            label_to_indices[label.item()] = (labels == label).nonzero(as_tuple=True)[0]
        
        anchor_indices = []
        positive_indices = []
        negative_indices = []
        
        # 生成一定数量的有效三元组
        target_count = min(10, batch_size // 3)
        attempts = 0
        max_attempts = target_count * 5
        
        while len(anchor_indices) < target_count and attempts < max_attempts:
            attempts += 1
            
            # 随机选择锚点
            anchor_idx = torch.randint(batch_size, (1,)).item()
            anchor_label = labels[anchor_idx].item()
            
            # 找正样本
            positive_candidates = label_to_indices[anchor_label]
            positive_candidates = positive_candidates[positive_candidates != anchor_idx]
            
            if len(positive_candidates) == 0:
                continue
            
            positive_idx = positive_candidates[torch.randint(len(positive_candidates), (1,))].item()
            
            # 找负样本
            negative_labels = [l for l in label_to_indices.keys() if l != anchor_label]
            if not negative_labels:
                continue
            
            negative_label = negative_labels[torch.randint(len(negative_labels), (1,))]
            negative_candidates = label_to_indices[negative_label]
            negative_idx = negative_candidates[torch.randint(len(negative_candidates), (1,))].item()
            
            anchor_indices.append(anchor_idx)
            positive_indices.append(positive_idx)
            negative_indices.append(negative_idx)
        
        if anchor_indices:
            self.logger.info(f"智能随机采样：生成 {len(anchor_indices)} 个有效三元组")
            return (torch.tensor(anchor_indices, device=device),
                    torch.tensor(positive_indices, device=device),
                    torch.tensor(negative_indices, device=device))
        else:
            # 极端情况：返回空
            empty_tensor = torch.tensor([], dtype=torch.long, device=device)
            return empty_tensor, empty_tensor, empty_tensor

class SemiHardMiner(TripletMiner):
    """半硬三元组挖掘器"""
    
    def __init__(self, margin: float = 0.3, squared: bool = False):
        self.margin = margin
        self.squared = squared
        self.logger = logging.getLogger(__name__)
    
    def mine_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """挖掘半硬三元组：d(a,p) < d(a,n) < d(a,p) + margin"""
        batch_size = embeddings.size(0)
        distance_matrix = self._compute_distance_matrix(embeddings)
        
        labels = labels.view(-1, 1)
        label_equal = labels == labels.t()
        
        anchor_indices = []
        positive_indices = []
        negative_indices = []
        
        for i in range(batch_size):
            positive_mask = label_equal[i] & (torch.arange(batch_size, device=embeddings.device) != i)
            negative_mask = ~label_equal[i]
            
            if not positive_mask.any() or not negative_mask.any():
                continue
            
            # 对每个正样本找半硬负样本
            for pos_idx in positive_mask.nonzero(as_tuple=True)[0]:
                pos_dist = distance_matrix[i, pos_idx]
                
                # 找到满足半硬条件的负样本
                neg_distances = distance_matrix[i][negative_mask]
                neg_indices = negative_mask.nonzero(as_tuple=True)[0]
                
                # 半硬条件：pos_dist < neg_dist < pos_dist + margin
                semi_hard_mask = (neg_distances > pos_dist) & (neg_distances < pos_dist + self.margin)
                
                if semi_hard_mask.any():
                    # 随机选择一个半硬负样本
                    valid_neg_indices = neg_indices[semi_hard_mask]
                    chosen_neg_idx = valid_neg_indices[torch.randint(len(valid_neg_indices), (1,))]
                    
                    anchor_indices.append(i)
                    positive_indices.append(pos_idx)
                    negative_indices.append(chosen_neg_idx)
        
        if not anchor_indices:
            return self._fallback_random_sampling(batch_size, labels, embeddings.device)
        
        return (torch.tensor(anchor_indices, device=embeddings.device),
                torch.tensor(positive_indices, device=embeddings.device),
                torch.tensor(negative_indices, device=embeddings.device))
    
    def _compute_distance_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.squared:
            distances = torch.cdist(embeddings, embeddings, p=2) ** 2
        else:
            distances = torch.cdist(embeddings, embeddings, p=2)
        return distances
    
    def _fallback_random_sampling(self, batch_size: int, labels: torch.Tensor, 
                                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_indices = torch.randperm(batch_size, device=device)[:batch_size//3]
        positive_indices = torch.randperm(batch_size, device=device)[:batch_size//3]
        negative_indices = torch.randperm(batch_size, device=device)[:batch_size//3]
        return anchor_indices, positive_indices, negative_indices

class AdaptiveMiner(TripletMiner):
    """自适应三元组挖掘器"""
    
    def __init__(self, margin: float = 0.3, hard_ratio: float = 0.2, 
                 semi_hard_ratio: float = 0.6, easy_ratio: float = 0.2):
        """
        自适应挖掘器，结合硬、半硬和简单三元组
        
        Args:
            margin: 边界值
            hard_ratio: 硬三元组比例
            semi_hard_ratio: 半硬三元组比例
            easy_ratio: 简单三元组比例
        """
        self.margin = margin
        self.hard_ratio = hard_ratio
        self.semi_hard_ratio = semi_hard_ratio
        self.easy_ratio = easy_ratio
        
        # 确保比例和为1
        total_ratio = hard_ratio + semi_hard_ratio + easy_ratio
        self.hard_ratio /= total_ratio
        self.semi_hard_ratio /= total_ratio
        self.easy_ratio /= total_ratio
        
        self.hard_miner = BatchHardMiner(margin)
        self.semi_hard_miner = SemiHardMiner(margin)
        self.logger = logging.getLogger(__name__)
    
    def mine_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor,
                     target_triplet_count: int = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """自适应挖掘不同难度的三元组"""
        
        if target_triplet_count is None:
            target_triplet_count = embeddings.size(0)
        
        # 计算各类三元组的目标数量
        hard_count = int(target_triplet_count * self.hard_ratio)
        semi_hard_count = int(target_triplet_count * self.semi_hard_ratio)
        easy_count = target_triplet_count - hard_count - semi_hard_count
        
        all_anchors = []
        all_positives = []
        all_negatives = []
        
        # 1. 挖掘硬三元组
        if hard_count > 0:
            hard_a, hard_p, hard_n = self.hard_miner.mine_triplets(embeddings, labels)
            if len(hard_a) > 0:
                # 限制数量
                if len(hard_a) > hard_count:
                    indices = torch.randperm(len(hard_a))[:hard_count]
                    hard_a, hard_p, hard_n = hard_a[indices], hard_p[indices], hard_n[indices]
                
                all_anchors.append(hard_a)
                all_positives.append(hard_p)
                all_negatives.append(hard_n)
                
                self.logger.debug(f"挖掘到 {len(hard_a)} 个硬三元组")
        
        # 2. 挖掘半硬三元组
        if semi_hard_count > 0:
            semi_a, semi_p, semi_n = self.semi_hard_miner.mine_triplets(embeddings, labels)
            if len(semi_a) > 0:
                if len(semi_a) > semi_hard_count:
                    indices = torch.randperm(len(semi_a))[:semi_hard_count]
                    semi_a, semi_p, semi_n = semi_a[indices], semi_p[indices], semi_n[indices]
                
                all_anchors.append(semi_a)
                all_positives.append(semi_p)
                all_negatives.append(semi_n)
                
                self.logger.debug(f"挖掘到 {len(semi_a)} 个半硬三元组")
        
        # 3. 补充简单三元组（随机采样）
        current_count = sum(len(a) for a in all_anchors)
        if current_count < target_triplet_count and easy_count > 0:
            remaining_count = min(easy_count, target_triplet_count - current_count)
            easy_a, easy_p, easy_n = self._mine_easy_triplets(embeddings, labels, remaining_count)
            
            if len(easy_a) > 0:
                all_anchors.append(easy_a)
                all_positives.append(easy_p)
                all_negatives.append(easy_n)
                
                self.logger.debug(f"补充了 {len(easy_a)} 个简单三元组")
        
        # 合并所有三元组
        if all_anchors:
            final_anchors = torch.cat(all_anchors)
            final_positives = torch.cat(all_positives)
            final_negatives = torch.cat(all_negatives)
        else:
            # 完全回退到随机采样
            final_anchors, final_positives, final_negatives = self._fallback_random_sampling(
                embeddings.size(0), labels, embeddings.device, target_triplet_count
            )
        
        return final_anchors, final_positives, final_negatives
    
    def _mine_easy_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor,
                          count: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """挖掘简单三元组（随机采样但确保标签正确）"""
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        anchors = []
        positives = []
        negatives = []
        
        # 创建标签映射
        unique_labels = torch.unique(labels)
        label_to_indices = {}
        for label in unique_labels:
            label_to_indices[label.item()] = (labels == label).nonzero(as_tuple=True)[0]
        
        attempts = 0
        max_attempts = count * 10
        
        while len(anchors) < count and attempts < max_attempts:
            attempts += 1
            
            # 随机选择锚点
            anchor_idx = torch.randint(batch_size, (1,)).item()
            anchor_label = labels[anchor_idx].item()
            
            # 找正样本
            positive_candidates = label_to_indices[anchor_label]
            positive_candidates = positive_candidates[positive_candidates != anchor_idx]
            
            if len(positive_candidates) == 0:
                continue
            
            positive_idx = positive_candidates[torch.randint(len(positive_candidates), (1,))].item()
            
            # 找负样本
            negative_labels = [l for l in label_to_indices.keys() if l != anchor_label]
            if not negative_labels:
                continue
            
            negative_label = np.random.choice(negative_labels)
            negative_candidates = label_to_indices[negative_label]
            negative_idx = negative_candidates[torch.randint(len(negative_candidates), (1,))].item()
            
            anchors.append(anchor_idx)
            positives.append(positive_idx)
            negatives.append(negative_idx)
        
        if anchors:
            return (torch.tensor(anchors, device=device),
                    torch.tensor(positives, device=device),
                    torch.tensor(negatives, device=device))
        else:
            return torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([], device=device)
    
    def _fallback_random_sampling(self, batch_size: int, labels: torch.Tensor, 
                                device: torch.device, count: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """完全随机采样作为最后回退"""
        self.logger.warning("使用完全随机采样作为回退方案")
        
        count = min(count, batch_size // 3)
        indices = torch.randperm(batch_size, device=device)
        
        return indices[:count], indices[count:2*count], indices[2*count:3*count]

class TripletMiningManager:
    """三元组挖掘管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化挖掘器
        mining_type = config.get('mining_type', 'adaptive')
        margin = config.get('margin', 0.5)
        
        if mining_type == 'hard':
            self.miner = BatchHardMiner(margin=margin)
        elif mining_type == 'semi_hard':
            self.miner = SemiHardMiner(margin=margin)
        elif mining_type == 'adaptive':
            self.miner = AdaptiveMiner(
                margin=margin,
                hard_ratio=config.get('hard_ratio', 0.3),
                semi_hard_ratio=config.get('semi_hard_ratio', 0.5),
                easy_ratio=config.get('easy_ratio', 0.2)
            )
        else:
            raise ValueError(f"未知的挖掘类型: {mining_type}")
        
        # 统计信息
        self.mining_stats = {
            'total_batches': 0,
            'successful_mining': 0,
            'fallback_count': 0,
            'average_triplets_per_batch': 0
        }
    
    def mine_triplets_from_batch(self, embeddings: torch.Tensor, labels: torch.Tensor,
                               **kwargs) -> TripletBatch:
        """从批次中挖掘三元组"""
        
        self.mining_stats['total_batches'] += 1
        
        try:
            # 执行挖掘
            anchor_indices, positive_indices, negative_indices = self.miner.mine_triplets(
                embeddings, labels, **kwargs
            )
            
            if len(anchor_indices) > 0:
                self.mining_stats['successful_mining'] += 1
                
                # 提取对应的嵌入和标签
                anchor_embeddings = embeddings[anchor_indices]
                positive_embeddings = embeddings[positive_indices]
                negative_embeddings = embeddings[negative_indices]
                
                anchor_labels = labels[anchor_indices]
                positive_labels = labels[positive_indices]
                negative_labels = labels[negative_indices]
                
                # 更新统计
                self.mining_stats['average_triplets_per_batch'] = (
                    (self.mining_stats['average_triplets_per_batch'] * (self.mining_stats['successful_mining'] - 1) + 
                     len(anchor_indices)) / self.mining_stats['successful_mining']
                )
                
                mining_info = {
                    'triplet_count': len(anchor_indices),
                    'mining_type': type(self.miner).__name__,
                    'success': True
                }
                
                return TripletBatch(
                    anchors=anchor_embeddings,
                    positives=positive_embeddings,
                    negatives=negative_embeddings,
                    anchor_labels=anchor_labels,
                    positive_labels=positive_labels,
                    negative_labels=negative_labels,
                    mining_info=mining_info
                )
            else:
                self.mining_stats['fallback_count'] += 1
                self.logger.warning("挖掘失败，返回空批次")
                
                return self._create_empty_batch(embeddings.device)
                
        except Exception as e:
            self.logger.error(f"三元组挖掘出错: {e}")
            self.mining_stats['fallback_count'] += 1
            return self._create_empty_batch(embeddings.device)
    
    def _create_empty_batch(self, device: torch.device) -> TripletBatch:
        """创建空的三元组批次"""
        empty_tensor = torch.empty(0, device=device)
        return TripletBatch(
            anchors=empty_tensor,
            positives=empty_tensor,
            negatives=empty_tensor,
            anchor_labels=empty_tensor,
            positive_labels=empty_tensor,
            negative_labels=empty_tensor,
            mining_info={'triplet_count': 0, 'success': False}
        )
    
    def get_mining_statistics(self) -> Dict:
        """获取挖掘统计信息"""
        stats = self.mining_stats.copy()
        
        if stats['total_batches'] > 0:
            stats['success_rate'] = stats['successful_mining'] / stats['total_batches']
            stats['fallback_rate'] = stats['fallback_count'] / stats['total_batches']
        else:
            stats['success_rate'] = 0.0
            stats['fallback_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.mining_stats = {
            'total_batches': 0,
            'successful_mining': 0,
            'fallback_count': 0,
            'average_triplets_per_batch': 0
        }


# 示例配置和使用
def demonstrate_hard_mining():
    """演示硬三元组挖掘"""
    
    # 模拟数据
    batch_size = 64
    embedding_dim = 128
    num_classes = 4
    
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 创建挖掘管理器
    config = {
        'mining_type': 'adaptive',
        'margin': 0.5,
        'hard_ratio': 0.4,
        'semi_hard_ratio': 0.4,
        'easy_ratio': 0.2
    }
    
    mining_manager = TripletMiningManager(config)
    
    # 执行挖掘
    triplet_batch = mining_manager.mine_triplets_from_batch(embeddings, labels)
    
    print(f"挖掘结果:")
    print(f"  三元组数量: {triplet_batch.mining_info['triplet_count']}")
    print(f"  挖掘类型: {triplet_batch.mining_info['mining_type']}")
    print(f"  成功状态: {triplet_batch.mining_info['success']}")
    
    # 显示统计
    stats = mining_manager.get_mining_statistics()
    print(f"\n挖掘统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    demonstrate_hard_mining()