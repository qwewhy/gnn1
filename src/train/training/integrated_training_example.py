# File: src/train/training/integrated_training_example.py
# 集成训练示例 / Integrated training example

import os
import sys
from pathlib import Path
import torch
import logging
import yaml
from typing import Dict, Any

# 添加项目路径到sys.path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.train.training.improved_train import ImprovedTrainer
from src.train.training.advanced_training_system import create_advanced_training_system
from src.train.training.hard_triplet_mining import TripletMiningManager
from src.train.models.improved_gat_encoder import MetricLearningGAT
from src.train.data_processing.pyg_dataset import PatchDataset
from src.train.data_processing.triplet_generator import TripletGenerator

class IntegratedTrainingSystem:
    """集成训练系统 / Integrated training system"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.config = self._load_config()
        
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化组件
        self.project_root = project_root
        self._setup_components()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载和验证配置"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            self.logger.warning(f"配置文件不存在: {self.config_path}")
            return self._create_default_config()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"配置文件加载失败: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """创建默认配置"""
        return {
            'training': {
                'learning_rate': 0.001,
                'weight_decay': 0.01,
                'margin': 0.5,
                'epochs': 50,
                'batch_size': 16,
                'patience': 10,
                'gradient_clip': 1.0,
                'checkpoint_dir': 'src/train/training/checkpoints/integrated',
                'num_triplets_train': 1000,
                'num_triplets_val': 200,
                'use_hard_mining': True,
                'use_advanced_system': False  # 控制是否使用高级系统
            },
            'model': {
                'anchor_in_channels': 8,
                'pattern_in_channels': 8,
                'hidden_channels': 64,
                'out_channels': 128,
                'num_heads': 4,
                'edge_dim': 3,
                'dropout': 0.2
            },
            'data': {
                'root': 'data/processed',
                'train_meshes': [
                    'model/stanford-bunny-retopo.obj',
                    'model/bull-retopo.obj'
                ],
                'val_meshes': [
                    'model/old-man-head-retopo.obj'
                ]
            },
            'mining': {
                'enabled': True,
                'mining_type': 'adaptive',
                'margin': 0.5,
                'hard_ratio': 0.3,
                'semi_hard_ratio': 0.5,
                'easy_ratio': 0.2,
                'candidate_multiplier': 3
            },
            'loss_weights': {
                'triplet': 1.0,
                'quality_tier': 0.3,
                'overall_regression': 0.5
            }
        }
    
    def _setup_components(self):
        """设置训练组件"""
        
        # 1. 创建数据集
        try:
            data_root = self.project_root / self.config['data']['root']
            self.patch_dataset = PatchDataset(root=str(data_root))
            self.logger.info(f"数据集加载成功，包含 {len(self.patch_dataset)} 个样本")
        except Exception as e:
            self.logger.error(f"数据集加载失败: {e}")
            self.patch_dataset = None
        
        # 2. 创建模型
        self.model = self._create_model()
        
        # 3. 准备三元组生成器（用于改进的三元组生成）
        self.triplet_generators = self._setup_triplet_generators()
        
    def _create_model(self) -> torch.nn.Module:
        """创建模型"""
        try:
            encoder_config = {
                'anchor_in_channels': self.config['model']['anchor_in_channels'],
                'pattern_in_channels': self.config['model']['pattern_in_channels'],
                'hidden_channels': self.config['model']['hidden_channels'],
                'out_channels': self.config['model']['out_channels'],
                'num_heads': self.config['model'].get('num_heads', 4),
                'edge_dim': self.config['model'].get('edge_dim', 3),
                'dropout': self.config['model'].get('dropout', 0.2)
            }
            
            model = MetricLearningGAT(encoder_config).to(self.device)
            self.logger.info("模型创建成功")
            return model
            
        except Exception as e:
            self.logger.error(f"模型创建失败: {e}")
            raise
    
    def _setup_triplet_generators(self):
        """设置三元组生成器"""
        generators = []
        
        if not self.patch_dataset:
            return generators
        
        # 硬挖掘配置
        hard_mining_config = self.config.get('mining', {})
        if self.config['training'].get('use_hard_mining', False):
            hard_mining_config['enabled'] = True
        else:
            hard_mining_config['enabled'] = False
        
        # 为每个网格文件创建生成器
        mesh_files = self.config['data']['train_meshes'] + self.config['data']['val_meshes']
        
        for mesh_path in mesh_files:
            full_path = self.project_root / mesh_path
            if full_path.exists():
                try:
                    generator = TripletGenerator(
                        str(full_path), 
                        self.patch_dataset,
                        hard_mining_config
                    )
                    generators.append(generator)
                    self.logger.info(f"三元组生成器创建成功: {mesh_path}")
                except Exception as e:
                    self.logger.warning(f"跳过文件 {mesh_path}: {e}")
            else:
                self.logger.warning(f"网格文件不存在: {full_path}")
        
        return generators
    
    def train_with_improved_system(self):
        """使用改进的训练系统（基于improved_train.py）"""
        self.logger.info("🚀 启动改进的训练系统")
        
        try:
            # 创建临时配置文件
            temp_config_path = self.project_root / 'temp_config.yaml'
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            # 使用ImprovedTrainer
            trainer = ImprovedTrainer(str(temp_config_path))
            
            # 如果有硬挖掘生成器，将其集成到训练器中
            if self.triplet_generators and self.config['training'].get('use_hard_mining', False):
                self._integrate_hard_mining_to_trainer(trainer)
            
            # 开始训练
            trainer.train()
            
            # 清理临时文件
            if temp_config_path.exists():
                temp_config_path.unlink()
            
            self.logger.info("✅ 改进的训练系统训练完成")
            return trainer
            
        except Exception as e:
            self.logger.error(f"改进的训练系统失败: {e}")
            raise
    
    def train_with_advanced_system(self):
        """使用高级训练系统（基于advanced_training_system.py）"""
        self.logger.info("🚀 启动高级训练系统")
        
        try:
            # 创建训练和验证数据集
            train_dataset = self._create_dataset_for_advanced_training('train')
            val_dataset = self._create_dataset_for_advanced_training('val')
            
            # 创建高级训练系统
            trainer = create_advanced_training_system(
                config_path=self.config_path,
                base_model=self.model.encoder,  # 使用基础编码器
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )
            
            # 开始训练
            results = trainer.train(num_epochs=self.config['training']['epochs'])
            
            self.logger.info("✅ 高级训练系统训练完成")
            return trainer, results
            
        except Exception as e:
            self.logger.error(f"高级训练系统失败: {e}")
            raise
    
    def _integrate_hard_mining_to_trainer(self, trainer):
        """将硬挖掘集成到改进的训练器中"""
        self.logger.info("🔧 集成硬三元组挖掘功能")
        
        # 为trainer添加硬挖掘功能
        mining_config = self.config.get('mining', {})
        trainer.hard_miner = TripletMiningManager(mining_config)
        
        # 修改训练器的三元组生成器
        if self.triplet_generators:
            for generator in self.triplet_generators:
                if hasattr(generator, 'hard_miner'):
                    self.logger.info(f"硬挖掘器已配置: {generator.hard_miner is not None}")
    
    def _create_dataset_for_advanced_training(self, split: str):
        """为高级训练系统创建数据集"""
        if split == 'train':
            return self.patch_dataset
        elif split == 'val':
            # 创建验证集（这里简化处理，实际应该分割数据集）
            if self.patch_dataset and len(self.patch_dataset) > 10:
                # 简单地使用后10%作为验证集
                val_size = len(self.patch_dataset) // 10
                indices = list(range(len(self.patch_dataset) - val_size, len(self.patch_dataset)))
                return torch.utils.data.Subset(self.patch_dataset, indices)
            return None
        return None
    
    def compare_training_systems(self):
        """对比两种训练系统"""
        self.logger.info("🔍 对比训练系统性能")
        
        results = {}
        
        # 1. 测试改进的训练系统
        try:
            self.logger.info("测试改进的训练系统...")
            improved_trainer = self.train_with_improved_system()
            results['improved'] = {
                'status': 'success',
                'trainer': improved_trainer
            }
        except Exception as e:
            self.logger.error(f"改进的训练系统测试失败: {e}")
            results['improved'] = {
                'status': 'failed',
                'error': str(e)
            }
        
        # 2. 测试高级训练系统（如果启用）
        if self.config['training'].get('use_advanced_system', False):
            try:
                self.logger.info("测试高级训练系统...")
                advanced_trainer, advanced_results = self.train_with_advanced_system()
                results['advanced'] = {
                    'status': 'success',
                    'trainer': advanced_trainer,
                    'results': advanced_results
                }
            except Exception as e:
                self.logger.error(f"高级训练系统测试失败: {e}")
                results['advanced'] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def demonstrate_hard_mining_integration(self):
        """演示硬挖掘集成"""
        self.logger.info("🎯 演示硬三元组挖掘集成")
        
        if not self.triplet_generators:
            self.logger.warning("没有可用的三元组生成器")
            return
        
        # 选择第一个生成器进行演示
        generator = self.triplet_generators[0]
        
        if hasattr(generator, 'hard_miner') and generator.hard_miner:
            self.logger.info("✅ 硬挖掘器已集成")
            
            # 测试硬挖掘功能
            try:
                # 生成一些测试三元组
                test_triplets = generator.generate_batch_triplets(
                    batch_size=8,
                    encoder_model=self.model
                )
                
                if test_triplets:
                    anchors, positives, negatives = test_triplets
                    self.logger.info(f"硬挖掘测试成功，生成了 {len(anchors)} 个三元组")
                    
                    # 显示挖掘统计
                    stats = generator.get_hard_mining_statistics()
                    if stats:
                        self.logger.info(f"挖掘统计: {stats}")
                else:
                    self.logger.warning("硬挖掘测试失败：未生成三元组")
                    
            except Exception as e:
                self.logger.error(f"硬挖掘测试出错: {e}")
        else:
            self.logger.info("使用几何三元组生成（硬挖掘未启用）")


def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # 创建集成训练系统
    try:
        # 尝试使用项目中的配置文件
        config_paths = [
            'configs/config.yaml',
            'configs/hard_mining_config.yaml',
            project_root / 'configs' / 'config.yaml'
        ]
        
        config_path = None
        for path in config_paths:
            if Path(path).exists():
                config_path = str(path)
                break
        
        if not config_path:
            logger.info("未找到配置文件，将使用默认配置")
            config_path = 'default_config.yaml'
        
        # 创建集成系统
        integrated_system = IntegratedTrainingSystem(config_path)
        
        # 演示硬挖掘集成
        integrated_system.demonstrate_hard_mining_integration()
        
        # 运行训练对比（可选）
        if input("是否运行训练对比测试？(y/n): ").lower() == 'y':
            results = integrated_system.compare_training_systems()
            
            logger.info("🏆 训练系统对比结果:")
            for system_name, result in results.items():
                logger.info(f"  {system_name}: {result['status']}")
                if result['status'] == 'failed':
                    logger.info(f"    错误: {result['error']}")
        
        logger.info("✅ 集成训练系统演示完成")
        
    except Exception as e:
        logger.error(f"集成训练系统演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()