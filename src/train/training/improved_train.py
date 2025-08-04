# File: src/train/training/improved_train.py
# 改进的训练脚本 / Improved training script

import os
# 解决 OpenMP 库冲突问题 / Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch  # 添加用于处理图数据批次的导入
import yaml
from pathlib import Path
import tqdm
# import wandb
from typing import Dict, List
import random
from collections import defaultdict

from src.train.models.improved_gat_encoder import MetricLearningGAT
from src.train.data_processing.pyg_dataset import PatchDataset
from src.train.data_processing.triplet_generator import TripletGenerator


class ImprovedTrainer:
    """
    改进的训练器，包含：
    - 难负例挖掘
    - 学习率调度
    - 早停机制
    - 详细的指标跟踪
    """

    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 确保关键数值配置的类型正确
        self._validate_config()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 修复路径问题 - 正确设置项目根目录
        self.project_root = Path(__file__).parent.parent.parent.parent  # 多加一个parent到真正的根目录
        print(f"🏠 项目根目录: {self.project_root.absolute()}")

        # 验证项目根目录是否正确
        expected_dirs = ['model', 'configs', 'data']
        missing_dirs = []
        for dir_name in expected_dirs:
            if not (self.project_root / dir_name).exists():
                missing_dirs.append(dir_name)

        if missing_dirs:
            print(f"⚠️  项目根目录可能不正确，缺少目录: {missing_dirs}")
            # 尝试其他可能的根目录
            current_path = Path(__file__).parent
            for i in range(5):  # 最多向上5级
                potential_root = current_path
                if all((potential_root / d).exists() for d in expected_dirs):
                    self.project_root = potential_root
                    print(f"🔄 修正项目根目录为: {self.project_root.absolute()}")
                    break
                current_path = current_path.parent

        # 修复路径问题 - 确保checkpoint_dir相对于项目根目录 / Fix path issue - ensure checkpoint_dir is relative to project root
        self.checkpoint_dir = self.project_root / self.config['training']['checkpoint_dir']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 检查点目录: {self.checkpoint_dir.absolute()}")

        # 初始化数据集
        self.setup_datasets()

        # 初始化模型
        self.setup_model()

        # 初始化训练组件
        self.setup_training_components()

        # 初始化指标跟踪
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch_metrics = defaultdict(list)

    def _validate_config(self):
        """验证并修正配置文件中的数据类型"""
        # 确保训练参数为正确的数值类型
        training_numeric_keys = {
            'learning_rate': float,
            'weight_decay': float,
            'margin': float,
            'l2_reg_weight': float,
            'gradient_clip': float,
            'epochs': int,
            'batch_size': int,
            'num_triplets_train': int,
            'num_triplets_val': int,
            'patience': int
        }

        for key, expected_type in training_numeric_keys.items():
            if key in self.config['training']:
                value = self.config['training'][key]
                if isinstance(value, str):
                    try:
                        self.config['training'][key] = expected_type(value)
                        print(f"Warning: 转换配置 '{key}' 从字符串 '{value}' 到 {expected_type.__name__}")
                    except ValueError:
                        print(f"Error: 无法转换配置 '{key}': {value}")
                        raise

    def collate_triplets(self, batch_list):
        """
        自定义的 collate 函数，用于处理三元组列表
        Custom collate function for handling triplet lists

        Args:
            batch_list: List of (anchor, positive, negative) tuples

        Returns:
            Tuple of (anchor_batch, positive_batch, negative_batch)
        """
        # 过滤掉生成失败的 None 值 / Filter out failed None values
        batch_list = [item for item in batch_list if item is not None]
        if not batch_list:
            return None, None, None  # 如果整个批次都是None，返回None

        # 将三元组列表分解为三个独立的列表 / Split triplet list into three separate lists
        anchors, positives, negatives = zip(*batch_list)

        # 使用 PyG 的 Batch.from_data_list 为每个组件创建批次
        # Use PyG's Batch.from_data_list to create batches for each component
        anchor_batch = Batch.from_data_list(list(anchors))
        positive_batch = Batch.from_data_list(list(positives))
        negative_batch = Batch.from_data_list(list(negatives))

        return anchor_batch, positive_batch, negative_batch

    def setup_datasets(self):
        """设置数据集和数据加载器"""
        # 使用项目根目录构建绝对路径 / Use project root to build absolute path
        data_root = self.project_root / self.config['data']['root']
        print(f"📂 数据根目录: {data_root.absolute()}")
        self.patch_dataset = PatchDataset(root=str(data_root))

        # 创建改进的三元组数据集
        class ImprovedTripletDataset(torch.utils.data.Dataset):
            def __init__(self, mesh_paths: List[str], patch_dataset, num_triplets: int):
                # 检查文件路径并过滤存在的文件
                valid_paths = []
                for path in mesh_paths:
                    if Path(path).exists():
                        valid_paths.append(path)
                        print(f"✅ 找到网格文件: {path}")
                    else:
                        print(f"❌ 网格文件不存在: {path}")

                if not valid_paths:
                    raise FileNotFoundError("没有找到有效的网格文件!")

                print(f"🎯 使用 {len(valid_paths)} 个网格文件生成三元组")

                # 为每个有效路径创建生成器
                self.generators = []
                for path in valid_paths:
                    try:
                        generator = TripletGenerator(path, patch_dataset)
                        self.generators.append(generator)
                        print(f"✅ 成功创建生成器: {Path(path).name}")
                    except Exception as e:
                        print(f"⚠️  跳过文件 {path}: {e}")

                if not self.generators:
                    raise RuntimeError("没有成功创建任何三元组生成器!")

                self.num_triplets = num_triplets
                self.patch_dataset = patch_dataset

            def __len__(self):
                return self.num_triplets

            def __getitem__(self, idx):
                # 随机选择一个生成器
                generator = random.choice(self.generators)
                triplet = None
                attempts = 0
                while triplet is None and attempts < 10:
                    triplet = generator.generate_triplet()
                    attempts += 1

                if triplet is None:
                    # 回退：生成随机三元组
                    return self._generate_random_triplet()

                return triplet

            def _generate_random_triplet(self):
                """生成随机三元组作为回退"""
                indices = random.sample(range(len(self.patch_dataset)), 3)
                return tuple(self.patch_dataset[i] for i in indices)

        # 修复路径构建 - 确保使用正确的项目根目录
        print("🔍 检查网格文件路径...")

        train_mesh_paths = []
        for relative_path in self.config['data']['train_meshes']:
            absolute_path = self.project_root / relative_path
            train_mesh_paths.append(str(absolute_path))
            print(f"训练网格: {absolute_path} - {'存在' if absolute_path.exists() else '不存在'}")

        val_mesh_paths = []
        for relative_path in self.config['data']['val_meshes']:
            absolute_path = self.project_root / relative_path
            val_mesh_paths.append(str(absolute_path))
            print(f"验证网格: {absolute_path} - {'存在' if absolute_path.exists() else '不存在'}")

        # 如果没有找到任何有效文件，尝试自动搜索
        if not any(Path(p).exists() for p in train_mesh_paths + val_mesh_paths):
            print("🔍 未找到配置的网格文件，自动搜索...")
            self._auto_find_mesh_files(train_mesh_paths, val_mesh_paths)

        self.train_dataset = ImprovedTripletDataset(
            train_mesh_paths,
            self.patch_dataset,
            self.config['training']['num_triplets_train']
        )

        self.val_dataset = ImprovedTripletDataset(
            val_mesh_paths,
            self.patch_dataset,
            self.config['training']['num_triplets_val']
        )

        # 数据加载器 / Data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],  # 从配置读取batch_size
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_triplets  # 使用自定义的collate函数
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],  # 从配置读取batch_size
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_triplets  # 使用自定义的collate函数
        )

        # 模式数据加载器（用于嵌入索引）
        self.pattern_loader = DataLoader(
            self.patch_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )

    def _auto_find_mesh_files(self, train_paths, val_paths):
        """自动搜索网格文件"""
        # 在项目根目录下搜索.obj文件
        obj_files = list(self.project_root.glob('**/*.obj'))

        if obj_files:
            print(f"🎯 找到 {len(obj_files)} 个.obj文件:")
            for obj_file in obj_files:
                rel_path = obj_file.relative_to(self.project_root)
                print(f"   {rel_path}")

            # 简单的文件分配策略
            if len(obj_files) >= 2:
                # 如果有多个文件，分别分配给训练和验证
                mid = len(obj_files) // 2
                train_paths.clear()
                val_paths.clear()
                train_paths.extend([str(f) for f in obj_files[:mid + 1]])
                val_paths.extend([str(f) for f in obj_files[mid:]])
                print(f"📊 自动分配: {len(train_paths)} 个训练文件, {len(val_paths)} 个验证文件")
            else:
                # 如果只有一个文件，同时用于训练和验证
                train_paths.clear()
                val_paths.clear()
                train_paths.append(str(obj_files[0]))
                val_paths.append(str(obj_files[0]))
                print(f"📊 使用唯一文件进行训练和验证: {obj_files[0].name}")
        else:
            print("❌ 未找到任何.obj文件!")
            raise FileNotFoundError("请将.obj文件放在项目目录中")

    def setup_model(self):
        """设置模型"""
        encoder_config = {
            'anchor_in_channels': self.config['model']['anchor_in_channels'],
            'pattern_in_channels': self.config['model']['pattern_in_channels'],
            'hidden_channels': self.config['model']['hidden_channels'],
            'out_channels': self.config['model']['out_channels'],
            'num_heads': self.config['model'].get('num_heads', 4),
            'edge_dim': self.config['model'].get('edge_dim', 3),
            'dropout': self.config['model'].get('dropout', 0.2)
        }

        self.model = MetricLearningGAT(encoder_config).to(self.device)

        # 尝试加载预训练权重
        pretrained_path = self.checkpoint_dir / 'pretrained_model.pt'
        if pretrained_path.exists():
            self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))
            print(f"Loaded pretrained model from {pretrained_path}")

    def setup_training_components(self):
        """设置训练组件"""
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.01)
        )

        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=self.config['training']['learning_rate'] * 0.01
        )

        # 损失函数参数
        self.margin = self.config['training']['margin']

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        metrics = defaultdict(float)

        progress_bar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        valid_batches = 0

        for batch_idx, (anchor, positive, negative) in enumerate(progress_bar):
            # 跳过无效批次 / Skip invalid batches
            if anchor is None or positive is None or negative is None:
                continue

            valid_batches += 1

            # 移动到设备
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            anchor_emb, positive_emb, negative_emb = self.model.forward_triplet(anchor, positive, negative)

            # 计算损失
            triplet_loss = self.model.compute_triplet_loss(
                anchor_emb, positive_emb, negative_emb, margin=self.margin
            )

            # 可选：添加正则化损失
            reg_loss = 0
            if self.config['training'].get('use_l2_reg', False):
                for param in self.model.parameters():
                    reg_loss += torch.norm(param, 2)
                # 确保 l2_reg_weight 是数值类型
                l2_weight = self.config['training'].get('l2_reg_weight', 1e-5)
                if isinstance(l2_weight, str):
                    l2_weight = float(l2_weight)
                reg_loss *= l2_weight

            total_loss = triplet_loss + reg_loss

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            self.optimizer.step()

            # 更新指标
            metrics['train_loss'] += total_loss.item()
            metrics['train_triplet_loss'] += triplet_loss.item()

            # 计算距离指标
            with torch.no_grad():
                pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb, p=2).mean()
                neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb, p=2).mean()
                metrics['train_pos_dist'] += pos_dist.item()
                metrics['train_neg_dist'] += neg_dist.item()

                # 准确率：正样本距离是否小于负样本距离
                correct = (pos_dist < neg_dist).float().mean()
                metrics['train_accuracy'] += correct.item()

            # 更新进度条
            progress_bar.set_postfix({
                'loss': total_loss.item(),
                'acc': correct.item()
            })

        # 计算平均指标
        if valid_batches > 0:
            for key in metrics:
                metrics[key] /= valid_batches
        else:
            print("⚠️  警告：没有有效的训练批次!")

        return dict(metrics)

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        metrics = defaultdict(float)

        progress_bar = tqdm.tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")
        valid_batches = 0

        with torch.no_grad():
            for anchor, positive, negative in progress_bar:
                # 跳过无效批次 / Skip invalid batches
                if anchor is None or positive is None or negative is None:
                    continue

                valid_batches += 1

                # 移动到设备
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                # 前向传播
                anchor_emb, positive_emb, negative_emb = self.model.forward_triplet(anchor, positive, negative)

                # 计算损失
                triplet_loss = self.model.compute_triplet_loss(
                    anchor_emb, positive_emb, negative_emb, margin=self.margin
                )

                metrics['val_loss'] += triplet_loss.item()

                # 计算距离指标
                pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb, p=2).mean()
                neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb, p=2).mean()
                metrics['val_pos_dist'] += pos_dist.item()
                metrics['val_neg_dist'] += neg_dist.item()

                # 准确率
                correct = (pos_dist < neg_dist).float().mean()
                metrics['val_accuracy'] += correct.item()

                progress_bar.set_postfix({
                    'loss': triplet_loss.item(),
                    'acc': correct.item()
                })

        # 计算平均指标
        if valid_batches > 0:
            for key in metrics:
                metrics[key] /= valid_batches
        else:
            print("⚠️  警告：没有有效的验证批次!")

        return dict(metrics)

    def compute_retrieval_metrics(self) -> Dict[str, float]:
        """计算检索指标（Recall@K）"""
        self.model.eval()

        # 构建模式嵌入索引
        pattern_embeddings = []
        pattern_labels = []

        with torch.no_grad():
            for batch in self.pattern_loader:
                batch = batch.to(self.device)
                embeddings = self.model.encoder(batch, input_type='pattern')
                pattern_embeddings.append(embeddings.cpu())
                # pattern_labels.extend(batch.quality.cpu().numpy()) # Assuming quality exists

        pattern_embeddings = torch.cat(pattern_embeddings, dim=0)
        # pattern_labels = np.array(pattern_labels) # Assuming quality exists

        # For now, let's assume we can't get labels easily, so we can't compute recall yet.
        # This part needs to be adapted once the dataset provides clear positive identities.
        # For the purpose of this refactoring, we will return an empty dict.

        return {}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # 保存最新检查点
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)

        # 如果是最佳模型，额外保存
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(self.model.state_dict(), best_path)
            print(f"Saved new best model with val_loss={metrics.get('val_loss', 'N/A'):.4f}")

    def train(self):
        """完整的训练流程"""
        # 可选：初始化wandb
        if self.config['training'].get('use_wandb', False):
            # wandb.init(project="quad-pattern-retrieval", config=self.config)
            print("wandb not installed. Skipping.")

        print(f"Starting training for {self.config['training']['epochs']} epochs...")

        for epoch in range(self.config['training']['epochs']):
            # 训练
            train_metrics = self.train_epoch(epoch)

            # 验证
            val_metrics = self.validate_epoch(epoch)

            # 计算检索指标（每5个epoch）
            if epoch % 5 == 0:
                retrieval_metrics = self.compute_retrieval_metrics()
                val_metrics.update(retrieval_metrics)

            # 合并指标
            all_metrics = {**train_metrics, **val_metrics}

            # 打印结果
            print(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}:")
            for key, value in all_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

            # 记录到wandb
            if self.config['training'].get('use_wandb', False):
                # wandb.log(all_metrics, step=epoch)
                pass

            # 学习率调度
            self.scheduler.step()

            # 早停检查
            current_val_loss = val_metrics.get('val_loss', float('inf'))
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, all_metrics, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, all_metrics, is_best=False)

                if self.patience_counter >= self.config['training'].get('patience', 10):
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        print("Training completed!")

        # 保存最终模型
        final_path = self.checkpoint_dir / 'final_model.pt'
        torch.save(self.model.state_dict(), final_path)
        print(f"Saved final model to {final_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='改进的GAT训练脚本')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')

    args = parser.parse_args()

    # 确保配置文件存在
    config_path = Path(args.config)
    if not config_path.exists():
        # 尝试从脚本所在目录向上查找配置文件
        script_dir = Path(__file__).parent
        for i in range(5):  # 最多向上5级目录
            potential_config = script_dir / args.config
            if potential_config.exists():
                config_path = potential_config
                break
            script_dir = script_dir.parent

    if not config_path.exists():
        print(f"❌ 配置文件不存在: {args.config}")
        print("💡 请确保配置文件在以下位置之一:")
        print("   - configs/config.yaml")
        print("   - 项目根目录/configs/config.yaml")
        return

    print(f"✅ 使用配置文件: {config_path.absolute()}")

    # 创建训练器并开始训练
    try:
        trainer = ImprovedTrainer(str(config_path))

        # 如果指定了恢复训练
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Resumed from checkpoint: {args.resume}")

        trainer.train()
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()