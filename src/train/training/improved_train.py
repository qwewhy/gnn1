# File: src/train/training/improved_train.py
# 训练主模块 / Main training module

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import yaml
from pathlib import Path
import sys
from collections import defaultdict
import argparse

# 设置项目路径，确保可以导入自定义模块
try:
    from src.common.path_manager import setup_project_environment
except ImportError:
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root / 'src'))
    from src.common.path_manager import setup_project_environment

# 导入项目模块
from src.train.models.improved_gat_encoder import MetricLearningGAT
from src.train.data_processing.pyg_dataset import PatchDataset
from src.train.data_processing.triplet_generator import TripletGenerator

class ImprovedTrainer:
    def __init__(self, config_path: str):
        # 1. 设置项目环境并获取路径管理器
        self.path_manager = setup_project_environment()
        self.project_root = self.path_manager.project_root
        
        # 2. 智能查找并加载配置文件
        config_file = self._find_config_file(config_path)
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        print(f"✅ 成功加载配置文件: {config_file}")

        # 3. 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ 使用设备: {self.device}")

        # 4. 设置检查点目录
        self.checkpoint_dir = self.project_root / self.config['training']['checkpoint_dir']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 检查点保存目录: {self.checkpoint_dir}")

        # 5. 初始化组件
        self.setup_datasets()
        self.setup_model()
        self.setup_training_components()
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _find_config_file(self, config_path: str) -> Path:
        """智能查找配置文件"""
        path_obj = Path(config_path)
        if path_obj.is_absolute() and path_obj.exists():
            return path_obj
        
        # 尝试在项目根目录的configs文件夹下查找
        config_in_configs = self.path_manager.get_config_path(path_obj.name)
        if config_in_configs.exists():
            return config_in_configs
            
        # 尝试直接在项目根目录查找
        config_in_root = self.project_root / path_obj.name
        if config_in_root.exists():
            return config_in_root
            
        raise FileNotFoundError(f"无法在任何标准位置找到配置文件: {config_path}")

    def setup_datasets(self):
        """设置数据集和数据加载器"""
        data_root = self.path_manager.data_dir
        print(f"📂 数据集根目录: {data_root}")
        
        self.patch_dataset = PatchDataset(root=str(data_root))
        
        # 解析训练和验证用的网格文件路径
        train_mesh_paths = [self.path_manager.get_model_path(p) for p in self.config['data']['train_meshes']]
        val_mesh_paths = [self.path_manager.get_model_path(p) for p in self.config['data']['val_meshes']]
        
        print(" MESH PATHS:", train_mesh_paths, val_mesh_paths)

        self.train_triplet_generator = TripletGenerator(train_mesh_paths, self.patch_dataset)
        self.val_triplet_generator = TripletGenerator(val_mesh_paths, self.patch_dataset)

        self.train_dataset = self.train_triplet_generator.generate_triplets(self.config['training']['num_triplets_train'])
        self.val_dataset = self.val_triplet_generator.generate_triplets(self.config['training']['num_triplets_val'])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False, collate_fn=self.collate_fn)
        
        print(f"📊 数据集设置完成: {len(self.train_dataset)} 训练样本, {len(self.val_dataset)} 验证样本")

    def setup_model(self):
        """设置GAT模型"""
        model_config = self.config['model']
        self.model = MetricLearningGAT(
            anchor_in_channels=model_config['anchor_in_channels'],
            pattern_in_channels=model_config['pattern_in_channels'],
            hidden_channels=model_config['hidden_channels'],
            out_channels=model_config['out_channels'],
            num_heads=model_config['num_heads'],
            edge_dim=model_config['edge_dim'],
            dropout=model_config['dropout']
        ).to(self.device)
        print("🤖 模型初始化完成")

    def setup_training_components(self):
        """设置优化器和损失函数"""
        train_config = self.config['training']
        self.optimizer = optim.AdamW(self.model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.loss_fn = torch.nn.TripletMarginLoss(margin=train_config['margin'])
        print("🛠️ 优化器与损失函数设置完成")

    @staticmethod
    def collate_fn(batch):
        """自定义数据整理函数"""
        anchors, positives, negatives = zip(*batch)
        return Batch.from_data_list(anchors), Batch.from_data_list(positives), Batch.from_data_list(negatives)

    def _run_epoch(self, loader, is_train=True):
        """运行一个epoch的训练或验证"""
        self.model.train(is_train)
        total_loss = 0.0
        
        for anchors, positives, negatives in loader:
            anchors = anchors.to(self.device)
            positives = positives.to(self.device)
            negatives = negatives.to(self.device)

            with torch.set_grad_enabled(is_train):
                anchor_embed = self.model(anchors)
                positive_embed = self.model(positives)
                negative_embed = self.model(negatives)
                loss = self.loss_fn(anchor_embed, positive_embed, negative_embed)

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                self.optimizer.step()
                
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def train(self):
        """主训练循环"""
        epochs = self.config['training']['epochs']
        patience = self.config['training']['patience']

        for epoch in range(epochs):
            train_loss = self._run_epoch(self.train_loader, is_train=True)
            val_loss = self._run_epoch(self.val_loader, is_train=False)
            
            print(f"Epoch {epoch+1}/{epochs} - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1

            self.save_checkpoint('latest_checkpoint.pt')

            if self.patience_counter >= patience:
                print("Early stopping triggered.")
                break
                
        self.save_checkpoint('final_model.pt')
        print("🏁 训练完成")

    def save_checkpoint(self, name):
        """保存模型检查点"""
        path = self.checkpoint_dir / name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss
        }, path)
        print(f"💾 模型已保存到: {path}")

def main():
    parser = argparse.ArgumentParser(description='运行GAT模型训练')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件的路径')
    args = parser.parse_args()

    try:
        trainer = ImprovedTrainer(config_path=args.config)
        trainer.train()
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
