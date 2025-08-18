# File: src/train/training/advanced_training_system.py
# 高级训练系统 / Advanced training system

# 修复OMP错误 - 必须在其他库导入之前设置
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from collections import defaultdict
import time
import yaml

from .hard_triplet_mining import TripletMiningManager, TripletBatch
from ..models.improved_gat_encoder import MetricLearningGAT
from ..evaluation.retrieval_metrics import RetrievalEvaluator


@dataclass
class TrainingMetrics:
    """训练指标数据类 / Training metrics dataclass"""
    epoch: int
    batch_idx: int

    # 损失指标 / Loss metrics
    total_loss: float
    triplet_loss: float
    classification_losses: Dict[str, float]
    regression_losses: Dict[str, float]

    # 准确率指标 / Accuracy metrics
    triplet_accuracy: float
    classification_accuracies: Dict[str, float]

    # 距离指标 / Distance metrics
    positive_distance: float
    negative_distance: float
    distance_margin: float

    # 挖掘指标 / Mining metrics
    mining_success_rate: float
    triplets_per_batch: int

    # 时间指标 / Time metrics
    batch_time: float
    mining_time: float


class MultiTaskLoss:
    """多任务损失函数 / Multi-task loss function"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 损失权重 / Loss weights - 简化配置，专注主要任务
        self.loss_weights = config.get('loss_weights', {
            'triplet': 1.0,               # 主要任务
            'overall_regression': 0.2,    # 降低权重
            'quality_tier': 0.1,          # 大幅降低权重
            'topology_binary': 0.0,       # 暂时禁用
            'distortion_binary': 0.0,     # 暂时禁用
            'valence_binary': 0.0,        # 暂时禁用
            'boundary_binary': 0.0,       # 暂时禁用
            'topology_regression': 0.0,   # 暂时禁用
            'geometric_regression': 0.0,  # 暂时禁用
            'regularity_regression': 0.0, # 暂时禁用
            'singularity_type': 0.0,      # 暂时禁用
            'boundary_type': 0.0          # 暂时禁用
        })

        # 损失函数 / Loss functions  
        self.triplet_loss = nn.TripletMarginLoss(margin=config.get('margin', 0.3))
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # 自适应权重（可选）/ Adaptive weights (optional)
        self.use_adaptive_weights = config.get('use_adaptive_weights', False)
        self.adaptive_weights = {}

    def compute_total_loss(self,
                           anchor_emb: torch.Tensor,
                           positive_emb: torch.Tensor,
                           negative_emb: torch.Tensor,
                           model_outputs: Dict[str, torch.Tensor],
                           targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算总损失 / Compute total loss"""

        losses = {}
        loss_values = {}
        device = anchor_emb.device

        # 1. 三元组损失 / Triplet loss
        if anchor_emb.numel() > 0 and positive_emb.numel() > 0 and negative_emb.numel() > 0:
            triplet_loss_val = self.triplet_loss(anchor_emb, positive_emb, negative_emb)
            losses['triplet'] = triplet_loss_val
            loss_values['triplet'] = triplet_loss_val.item()
        else:
            losses['triplet'] = torch.tensor(0.0, device=device)
            loss_values['triplet'] = 0.0

        # 2. 二元分类损失 / Binary classification losses
        binary_tasks = ['topology_binary', 'distortion_binary', 'valence_binary', 'boundary_binary']
        for task in binary_tasks:
            if task in model_outputs and task in targets:
                loss_val = self.bce_loss(model_outputs[task], targets[task].float())
                losses[task] = loss_val
                loss_values[task] = loss_val.item()

        # 3. 回归损失 / Regression losses
        regression_tasks = ['topology_regression', 'geometric_regression', 'regularity_regression',
                            'overall_regression']
        for task in regression_tasks:
            if task in model_outputs and task in targets:
                loss_val = self.mse_loss(model_outputs[task], targets[task].float())
                losses[task] = loss_val
                loss_values[task] = loss_val.item()

        # 4. 多类分类损失 / Multi-class classification losses
        classification_tasks = ['quality_tier', 'singularity_type', 'boundary_type']
        for task in classification_tasks:
            if task in model_outputs and task in targets:
                loss_val = self.ce_loss(model_outputs[task], targets[task].long())
                losses[task] = loss_val
                loss_values[task] = loss_val.item()

        # 5. 计算加权总损失 / Compute weighted total loss
        total_loss = torch.tensor(0.0, device=device)

        for task, loss_val in losses.items():
            weight = self.loss_weights.get(task, 1.0)

            # 自适应权重调整 / Adaptive weight adjustment
            if self.use_adaptive_weights and task in self.adaptive_weights:
                weight *= self.adaptive_weights[task]

            total_loss += weight * loss_val

        loss_values['total'] = total_loss.item()

        return total_loss, loss_values


class AdvancedGATModel(nn.Module):
    """增强的GAT模型，支持多任务学习 / Enhanced GAT model with multi-task learning"""

    def __init__(self, base_encoder, config: Dict):
        super().__init__()
        self.base_encoder = base_encoder
        self.config = config

        embedding_dim = config['model']['out_channels']

        # 多任务头 / Multi-task heads
        self.task_heads = nn.ModuleDict({
            # 二元分类头 / Binary classification heads
            'topology_binary': nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            ),
            'distortion_binary': nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            ),
            'valence_binary': nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            ),
            'boundary_binary': nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            ),

            # 回归头 / Regression heads
            'topology_regression': nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ),
            'geometric_regression': nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ),
            'regularity_regression': nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ),
            'overall_regression': nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ),

            # 多类分类头 / Multi-class classification heads
            'quality_tier': nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 4)  # excellent, good, fair, poor
            ),
            'singularity_type': nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 3)  # none, regular, irregular
            ),
            'boundary_type': nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 3)  # smooth, feature, irregular
            )
        })

    def forward(self, data, input_type: str = 'pattern') -> Dict[str, torch.Tensor]:
        """前向传播 / Forward propagation"""
        # 获取基础嵌入 / Get base embeddings
        embeddings = self.base_encoder(data, input_type=input_type)

        # 计算各任务输出 / Compute outputs for each task
        outputs = {'embeddings': embeddings}

        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(embeddings)

        return outputs


class AdvancedTrainer:
    """高级训练器 / Advanced trainer"""

    def __init__(self, config: Dict, model, train_dataset, val_dataset, device: torch.device):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.logger = logging.getLogger(__name__)

        # 训练组件 / Training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.loss_computer = MultiTaskLoss(config)

        # 硬三元组挖掘器 / Hard triplet miner
        mining_config = config.get('mining', {
            'enabled': True,
            'mining_type': 'adaptive',
            'margin': 0.3,
            'hard_ratio': 0.2,
            'semi_hard_ratio': 0.6,
            'easy_ratio': 0.2
        })
        self.triplet_miner = TripletMiningManager(mining_config)
        
        # 检索评估器 / Retrieval evaluator
        evaluation_config = config.get('evaluation', {})
        k_values = evaluation_config.get('retrieval', {}).get('k_values', [1, 5, 10, 20])
        self.retrieval_evaluator = RetrievalEvaluator(k_values=k_values)

        # 指标跟踪 / Metrics tracking
        self.training_history = []
        self.validation_history = []
        self.retrieval_history = []  # 新增：检索性能历史
        self.best_metrics = {}
        self.best_map = 0.0  # 新增：最佳mAP记录

        # 训练配置 / Training configuration
        training_config = config.get('training', {})
        
        # 早停 / Early stopping
        self.patience = training_config.get('patience', 10)
        self.patience_counter = 0
        self.best_loss = float('inf')

        # 检查点管理 / Checkpoint management
        self.checkpoint_dir = Path(training_config.get('checkpoint_dir', 'src/train/training/checkpoints/advanced'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 数据加载器 / Data loaders
        # 检查数据集类型并选择适合的 DataLoader
        batch_size = training_config.get('batch_size', 32)
        self.train_loader = self._create_data_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # 修改后 - 明确不使用验证集
        self.val_loader = None
        self.logger.info("🚫 已禁用验证集，使用全部数据训练")

    def _create_data_loader(self, dataset, batch_size: int, shuffle: bool):
        """创建适合数据类型的 DataLoader / Create appropriate DataLoader for data type"""
        # 检查数据集中的第一个样本类型
        if len(dataset) > 0:
            sample = dataset[0]
            # 如果是 PyG Data 对象，使用 PyG DataLoader
            if hasattr(sample, 'x') and hasattr(sample, 'edge_index'):
                self.logger.info("使用 PyTorch Geometric DataLoader")
                return PyGDataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=0
                )
            # 如果是三元组数据（tuple），使用普通 DataLoader
            elif isinstance(sample, (tuple, list)):
                self.logger.info("使用标准 PyTorch DataLoader (三元组数据)")
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=0
                )

        # 默认使用 PyG DataLoader
        self.logger.info("默认使用 PyTorch Geometric DataLoader")
        return PyGDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )

    def _setup_optimizer(self):
        """设置优化器 / Setup optimizer"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'AdamW')

        if optimizer_type == 'AdamW':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=float(optimizer_config.get('lr', 0.001)),  # 确保转换为float
                weight_decay=float(optimizer_config.get('weight_decay', 0.01))
            )
        elif optimizer_type == 'Adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=float(optimizer_config.get('lr', 0.001)),
                weight_decay=float(optimizer_config.get('weight_decay', 0.01))
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    def _setup_scheduler(self):
        """设置学习率调度器 / Setup learning rate scheduler"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'CosineAnnealingLR')

        # 获取训练epochs数，确保类型正确
        total_epochs = int(self.config.get('training', {}).get('epochs', 100))

        if scheduler_type == 'CosineAnnealingLR':
            eta_min = scheduler_config.get('eta_min', 1e-6)
            # 确保eta_min是float类型
            if isinstance(eta_min, str):
                eta_min = float(eta_min)
            elif isinstance(eta_min, (int, float)):
                eta_min = float(eta_min)
            else:
                eta_min = 1e-6

            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=eta_min
            )
        elif scheduler_type == 'StepLR':
            step_size = int(scheduler_config.get('step_size', 30))
            gamma = float(scheduler_config.get('gamma', 0.1))
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            factor = float(scheduler_config.get('factor', 0.5))
            patience = int(scheduler_config.get('patience', 5))
            threshold = float(scheduler_config.get('threshold', 1e-4))
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                threshold=threshold
            )
        else:
            return None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch / Train one epoch"""
        self.model.train()
        epoch_metrics = defaultdict(list)

        for batch_idx, batch_data in enumerate(self.train_loader):
            batch_start_time = time.time()

            # 准备数据 / Prepare data
            if hasattr(batch_data, 'to'):
                batch_data = batch_data.to(self.device)
            else:
                # 如果是三元组数据
                anchor, positive, negative = batch_data
                if anchor is None or positive is None or negative is None:
                    continue
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

            # 前向传播 / Forward pass
            if hasattr(batch_data, 'to'):
                # 单个批次数据
                model_outputs = self.model(batch_data, input_type='pattern')
                embeddings = model_outputs['embeddings']

                # 准备标签 / Prepare labels
                targets = self._extract_targets_from_batch(batch_data)

                # 三元组挖掘 / Triplet mining
                mining_start_time = time.time()
                labels = batch_data.quality if hasattr(batch_data, 'quality') else torch.arange(len(embeddings))
                triplet_batch = self.triplet_miner.mine_triplets_from_batch(embeddings, labels)
                mining_time = time.time() - mining_start_time

                # 获取三元组嵌入
                if triplet_batch.anchors.numel() > 0:
                    anchor_emb = triplet_batch.anchors
                    positive_emb = triplet_batch.positives
                    negative_emb = triplet_batch.negatives
                else:
                    anchor_emb = torch.empty(0, embeddings.size(1), device=self.device)
                    positive_emb = torch.empty(0, embeddings.size(1), device=self.device)
                    negative_emb = torch.empty(0, embeddings.size(1), device=self.device)
            else:
                # 三元组数据
                anchor_outputs = self.model(anchor, input_type='anchor')
                positive_outputs = self.model(positive, input_type='pattern')
                negative_outputs = self.model(negative, input_type='pattern')

                anchor_emb = anchor_outputs['embeddings']
                positive_emb = positive_outputs['embeddings']
                negative_emb = negative_outputs['embeddings']

                model_outputs = positive_outputs  # 使用正样本的输出作为主要输出
                targets = self._extract_targets_from_batch(positive)
                mining_time = 0  # 不需要挖掘时间

            # 计算损失 / Compute loss
            total_loss, loss_values = self.loss_computer.compute_total_loss(
                anchor_emb, positive_emb, negative_emb, model_outputs, targets
            )

            # 反向传播 / Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪 / Gradient clipping
            training_config = self.config.get('training', {})
            gradient_clip = training_config.get('gradient_clip', 0)
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    gradient_clip
                )

            self.optimizer.step()

            # 计算指标 / Compute metrics
            batch_time = time.time() - batch_start_time
            metrics = self._compute_batch_metrics(
                anchor_emb, positive_emb, negative_emb, model_outputs, targets,
                loss_values, batch_time, mining_time
            )

            # 记录指标 / Record metrics
            for key, value in metrics.items():
                epoch_metrics[key].append(value)

            # 日志输出 / Log output
            training_config = self.config.get('training', {})
            if batch_idx % training_config.get('log_interval', 50) == 0:
                triplet_count = len(anchor_emb) if anchor_emb.numel() > 0 else 0
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Loss={total_loss:.4f}, "
                    f"Triplets={triplet_count}, "
                    f"Mining_time={mining_time:.3f}s"
                )

        # 计算epoch平均指标 / Compute epoch average metrics
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        avg_metrics['epoch'] = epoch

        return avg_metrics

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch / Validate one epoch"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        epoch_metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.val_loader):
                # 准备数据 / Prepare data
                if hasattr(batch_data, 'to'):
                    batch_data = batch_data.to(self.device)
                else:
                    anchor, positive, negative = batch_data
                    if anchor is None or positive is None or negative is None:
                        continue
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)

                # 前向传播 / Forward pass
                if hasattr(batch_data, 'to'):
                    model_outputs = self.model(batch_data, input_type='pattern')
                    embeddings = model_outputs['embeddings']
                    targets = self._extract_targets_from_batch(batch_data)

                    # 三元组挖掘 / Triplet mining
                    labels = batch_data.quality if hasattr(batch_data, 'quality') else torch.arange(len(embeddings))
                    triplet_batch = self.triplet_miner.mine_triplets_from_batch(embeddings, labels)

                    if triplet_batch.anchors.numel() > 0:
                        anchor_emb = triplet_batch.anchors
                        positive_emb = triplet_batch.positives
                        negative_emb = triplet_batch.negatives
                    else:
                        anchor_emb = torch.empty(0, embeddings.size(1), device=self.device)
                        positive_emb = torch.empty(0, embeddings.size(1), device=self.device)
                        negative_emb = torch.empty(0, embeddings.size(1), device=self.device)
                else:
                    anchor_outputs = self.model(anchor, input_type='anchor')
                    positive_outputs = self.model(positive, input_type='pattern')
                    negative_outputs = self.model(negative, input_type='pattern')

                    anchor_emb = anchor_outputs['embeddings']
                    positive_emb = positive_outputs['embeddings']
                    negative_emb = negative_outputs['embeddings']

                    model_outputs = positive_outputs
                    targets = self._extract_targets_from_batch(positive)

                # 计算损失 / Compute loss
                total_loss, loss_values = self.loss_computer.compute_total_loss(
                    anchor_emb, positive_emb, negative_emb, model_outputs, targets
                )

                # 计算指标 / Compute metrics
                metrics = self._compute_batch_metrics(
                    anchor_emb, positive_emb, negative_emb, model_outputs, targets,
                    loss_values, 0, 0  # 验证时不计算时间
                )

                # 记录指标 / Record metrics
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)

        # 计算epoch平均指标 / Compute epoch average metrics
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        avg_metrics['epoch'] = epoch

        return avg_metrics

    def _extract_targets_from_batch(self, batch_data) -> Dict[str, torch.Tensor]:
        """从批次数据中提取目标标签 / Extract target labels from batch data"""
        targets = {}

        # 如果有多任务目标 / If multi-task targets exist
        if hasattr(batch_data, 'multi_task_targets'):
            for task_name, target_tensor in batch_data.multi_task_targets.items():
                targets[task_name] = target_tensor.to(self.device)

        # 基础质量标签 / Basic quality labels
        if hasattr(batch_data, 'quality'):
            targets['quality_tier'] = batch_data.quality.to(self.device)

        return targets

    def _compute_batch_metrics(self,
                               anchor_emb: torch.Tensor,
                               positive_emb: torch.Tensor,
                               negative_emb: torch.Tensor,
                               model_outputs: Dict[str, torch.Tensor],
                               targets: Dict[str, torch.Tensor],
                               loss_values: Dict[str, float],
                               batch_time: float,
                               mining_time: float) -> Dict[str, float]:
        """计算批次指标 / Compute batch metrics"""
        metrics = {}

        # 损失指标 / Loss metrics
        metrics.update(loss_values)

        # 三元组指标 / Triplet metrics
        if anchor_emb.numel() > 0 and positive_emb.numel() > 0 and negative_emb.numel() > 0:
            pos_dist = F.pairwise_distance(anchor_emb, positive_emb).mean()
            neg_dist = F.pairwise_distance(anchor_emb, negative_emb).mean()

            metrics['positive_distance'] = pos_dist.item()
            metrics['negative_distance'] = neg_dist.item()
            metrics['distance_margin'] = (neg_dist - pos_dist).item()
            metrics['triplet_accuracy'] = (pos_dist < neg_dist).float().mean().item()
        else:
            metrics['positive_distance'] = 0
            metrics['negative_distance'] = 0
            metrics['distance_margin'] = 0
            metrics['triplet_accuracy'] = 0

        # 分类准确率 / Classification accuracy
        for task in ['quality_tier', 'singularity_type', 'boundary_type']:
            if task in model_outputs and task in targets:
                predictions = torch.argmax(model_outputs[task], dim=1)
                accuracy = (predictions == targets[task]).float().mean()
                metrics[f'{task}_accuracy'] = accuracy.item()

        # 二元分类准确率 / Binary classification accuracy
        for task in ['topology_binary', 'distortion_binary', 'valence_binary', 'boundary_binary']:
            if task in model_outputs and task in targets:
                predictions = torch.sigmoid(model_outputs[task]) > 0.5
                accuracy = (predictions.squeeze() == targets[task]).float().mean()
                metrics[f'{task}_accuracy'] = accuracy.item()

        # 挖掘指标 / Mining metrics
        mining_stats = self.triplet_miner.get_mining_statistics()
        metrics['mining_success_rate'] = mining_stats.get('success_rate', 0)
        metrics['triplets_per_batch'] = len(anchor_emb) if anchor_emb.numel() > 0 else 0

        # 时间指标 / Time metrics
        metrics['batch_time'] = batch_time
        metrics['mining_time'] = mining_time

        return metrics
    
    def evaluate_retrieval_performance(self, epoch: int) -> Dict[str, float]:
        """评估检索性能 / Evaluate retrieval performance"""
        try:
            self.logger.info(f"🎯 开始第 {epoch} 轮检索性能评估...")
            
            # 使用训练数据集评估（因为没有验证集）
            # 注意：需要传递base_encoder而不是整个AdvancedGATModel
            base_model = self.model.base_encoder if hasattr(self.model, 'base_encoder') else self.model
            
            retrieval_metrics = self.retrieval_evaluator.evaluate_model_retrieval(
                model=base_model,
                dataset=self.train_dataset,
                device=self.device,
                max_samples=500,  # 限制评估样本数以节省时间
                batch_size=16
            )
            
            # 添加epoch信息
            retrieval_metrics['epoch'] = epoch
            
            # 记录到历史
            self.retrieval_history.append(retrieval_metrics)
            
            # 更新最佳mAP
            current_map = retrieval_metrics.get('mAP', 0.0)
            if current_map > self.best_map:
                self.best_map = current_map
                self.logger.info(f"🏆 新的最佳mAP: {current_map:.4f}")
            
            # 打印核心指标
            self.logger.info(f"检索评估结果 - mAP: {current_map:.4f}, "
                           f"P@5: {retrieval_metrics.get('precision@5', 0):.4f}, "
                           f"P@10: {retrieval_metrics.get('precision@10', 0):.4f}")
            
            return retrieval_metrics
            
        except Exception as e:
            self.logger.error(f"检索评估失败: {e}")
            return {'mAP': 0.0, 'error': str(e), 'epoch': epoch}

    def train(self, num_epochs: int = None) -> Dict[str, Any]:
        """完整训练流程 / Complete training process"""

        if num_epochs is None:
            num_epochs = int(self.config.get('training', {}).get('epochs', 100))

        self.logger.info(f"开始训练，共 {num_epochs} 个epoch")
        
        # 评估配置
        evaluation_config = self.config.get('evaluation', {})
        eval_frequency = evaluation_config.get('eval_frequency', 5)

        for epoch in range(num_epochs):
            # 训练 / Training
            train_metrics = self.train_epoch(epoch)
            self.training_history.append(train_metrics)

            # 修改后 - 跳过验证
            val_metrics = {}  # 空的验证指标
            self.validation_history.append({})  # 保持结构一致
            
            # 检索性能评估 / Retrieval performance evaluation
            retrieval_metrics = {}
            if epoch % eval_frequency == 0 or epoch == num_epochs - 1:
                retrieval_metrics = self.evaluate_retrieval_performance(epoch)

            # 学习率调度 / Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # 修改后 - 使用训练损失
                    self.scheduler.step(train_metrics['total'])  # 基于训练损失
                else:
                    # 其他调度器直接step
                    self.scheduler.step()

            # 修改后 - 基于训练损失进行早停
            current_val_loss = train_metrics['total']  # 直接使用训练损失

            if current_val_loss < self.best_loss:
                self.best_loss = current_val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
                self._save_checkpoint(epoch, train_metrics, val_metrics, is_best=False)

            # 输出epoch总结 / Output epoch summary
            map_info = ""
            if retrieval_metrics and 'mAP' in retrieval_metrics:
                map_info = f", mAP={retrieval_metrics['mAP']:.4f}"
            
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['total']:.4f}, "
                f"Val Loss={current_val_loss:.4f}, "
                f"LR={self.optimizer.param_groups[0]['lr']:.6f}, "
                f"Patience={self.patience_counter}/{self.patience}"
                f"{map_info}"
            )

            # 早停 / Early stopping
            if self.patience_counter >= self.patience:
                self.logger.info(f"早停触发，在第 {epoch} 轮停止训练")
                break

        # 返回训练历史 / Return training history
        return {
            'training_history': self.training_history,
            'validation_history': self.validation_history,
            'retrieval_history': self.retrieval_history,  # 新增检索历史
            'best_metrics': self.best_metrics,
            'best_map': self.best_map,  # 新增最佳mAP
            'final_epoch': epoch
        }

    def _save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Dict, is_best: bool = False):
        """保存检查点 / Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config,
            'training_history': self.training_history,
            'validation_history': self.validation_history,
            'retrieval_history': self.retrieval_history,  # 新增检索历史
            'best_map': self.best_map  # 新增最佳mAP
        }

        # 保存最新检查点 / Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)

        # 保存最佳模型 / Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(self.model.state_dict(), best_path)

            # 保存完整的最佳检查点 / Save complete best checkpoint
            best_checkpoint_path = self.checkpoint_dir / 'best_checkpoint.pt'
            torch.save(checkpoint, best_checkpoint_path)

            self.best_metrics = val_metrics if val_metrics else train_metrics
            
            # 修复格式化错误：处理验证损失显示
            val_loss = val_metrics.get('total') if val_metrics else None
            if val_loss is not None:
                self.logger.info(f"保存最佳模型，验证损失: {val_loss:.4f}")
            else:
                self.logger.info(f"保存最佳模型，训练损失: {train_metrics.get('total', 0):.4f}")


def create_advanced_training_system(config_path: str, base_model, train_dataset, val_dataset=None):
    """创建高级训练系统 / Create advanced training system"""

    # 加载配置 / Load configuration
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # 默认配置 / Default configuration
        config = {
            'model': {
                'out_channels': 128
            },
            'mining': {
                'enabled': True,
                'mining_type': 'adaptive',
                'margin': 0.5,
                'hard_ratio': 0.3,
                'semi_hard_ratio': 0.5,
                'easy_ratio': 0.2
            },
            'loss_weights': {
                'triplet': 1.0,
                'overall_regression': 0.5,
                'quality_tier': 0.3
            },
            'optimizer': {
                'type': 'AdamW',
                'lr': 0.001,
                'weight_decay': 0.01
            },
            'scheduler': {
                'type': 'CosineAnnealingLR',
                'eta_min': 1e-6
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'patience': 15,
                'gradient_clip': 1.0,
                'log_interval': 50,
                'checkpoint_dir': 'src/train/training/checkpoints/advanced'
            }
        }

    # 设备设置 / Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建增强模型 / Create enhanced model
    enhanced_model = AdvancedGATModel(base_model, config)
    enhanced_model.to(device)

    # 创建训练器 / Create trainer
    trainer = AdvancedTrainer(config, enhanced_model, train_dataset, val_dataset, device)

    print("🚀 高级训练系统配置完成")
    print(f"📊 使用挖掘策略: {config['mining']['mining_type']}")
    print(f"⚖️  损失权重: {config['loss_weights']}")
    print(f"🔧 优化器: {config['optimizer']['type']}")
    print(f"📱 设备: {device}")

    return trainer


# 使用示例 / Usage example
def demonstrate_advanced_training():
    """演示高级训练系统 / Demonstrate advanced training system"""

    print("🎯 高级训练系统演示")

    # 这里需要替换为实际的模型和数据集
    # Here you need to replace with actual model and datasets

    config_example = {
        'model': {'out_channels': 128},
        'mining': {
            'enabled': True,
            'mining_type': 'adaptive',
            'margin': 0.5,
            'hard_ratio': 0.4,
            'semi_hard_ratio': 0.4,
            'easy_ratio': 0.2
        },
        'loss_weights': {
            'triplet': 1.0,
            'quality_tier': 0.3,
            'overall_regression': 0.5
        },
        'optimizer': {
            'type': 'AdamW',
            'lr': 0.001,
            'weight_decay': 0.01
        },
        'training': {
            'epochs': 10,
            'batch_size': 16,
            'patience': 5,
            'log_interval': 10
        }
    }

    print("配置示例:")
    print(json.dumps(config_example, indent=2, ensure_ascii=False))

    return config_example


if __name__ == '__main__':
    # 配置日志 / Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 运行演示 / Run demonstration
    config = demonstrate_advanced_training()
    print("\n✅ 高级训练系统准备就绪！")
    print("💡 请使用 create_advanced_training_system() 函数来创建完整的训练系统")