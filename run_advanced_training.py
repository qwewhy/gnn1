# File: run_advanced_training.py
# 高级训练系统启动脚本

# 修复OMP错误 - 必须在其他库导入之前设置
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# 现在可以导入项目模块
from src.train.training.advanced_training_system import create_advanced_training_system, demonstrate_advanced_training
from src.train.models.improved_gat_encoder import MetricLearningGAT
from src.train.data_processing.pyg_dataset import PatchDataset
from src.common.path_manager import setup_project_environment
import torch
import yaml
import logging


def main():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # 设置项目环境
        path_manager = setup_project_environment()
        logger.info(f"✅ 项目根目录: {path_manager.project_root}")

        # 检查配置文件
        config_path = path_manager.get_config_path('advanced_training_config.yaml')
        if not config_path.exists():
            logger.warning(f"⚠️ 配置文件不存在: {config_path}")
            logger.info("📝 使用默认配置运行演示")

            # 运行演示
            demonstrate_advanced_training()
            return

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ 成功加载配置: {config_path}")

        # 检查数据集
        dataset_path = path_manager.data_processed_dir
        if not dataset_path.exists():
            logger.error(f"❌ 数据集目录不存在: {dataset_path}")
            logger.info("💡 请先运行: python src/train/data_processing/populate_db.py")
            return

        # 加载数据集
        try:
            train_dataset = PatchDataset(root=str(dataset_path))
            logger.info(f"✅ 训练数据集加载成功: {len(train_dataset)} 个样本")

            # 修改后 - 全部用于训练
            val_dataset = None  # 不使用验证集
            # train_dataset 保持不变，使用全部数据
            logger.info(f"📊 使用全部数据训练: {len(train_dataset)} 个样本")

        except Exception as e:
            logger.error(f"❌ 数据集加载失败: {e}")
            logger.info("💡 请先运行: python src/train/data_processing/populate_db.py")
            return

        # 创建基础编码器
        if len(train_dataset) > 0:
            sample_data = train_dataset[0]
            model_config = config.get('model', {})

            encoder_config = {
                'anchor_in_channels': model_config.get('anchor_in_channels', 8),
                'pattern_in_channels': model_config.get('pattern_in_channels', 8),
                'hidden_channels': model_config.get('hidden_channels', 64),
                'out_channels': model_config.get('out_channels', 128),
                'num_heads': model_config.get('num_heads', 4),
                'edge_dim': model_config.get('edge_dim', 3),
                'dropout': model_config.get('dropout', 0.2)
            }

            base_model = MetricLearningGAT(encoder_config)
            logger.info("✅ 基础模型创建成功")
        else:
            logger.error("❌ 数据集为空，无法创建模型")
            return

        # 创建高级训练系统
        trainer = create_advanced_training_system(
            config_path=str(config_path),
            base_model=base_model.encoder,  # 使用编码器部分
            train_dataset=train_dataset,
            val_dataset=val_dataset
        )

        # 开始训练
        logger.info("🚀 开始高级训练...")
        training_config = config.get('training', {})
        num_epochs = training_config.get('epochs', 10)
        results = trainer.train(num_epochs=num_epochs)

        logger.info("✅ 训练完成!")
        logger.info(f"📈 最终训练轮数: {results['final_epoch']}")
        logger.info(f"📊 最佳指标: {results['best_metrics']}")

    except Exception as e:
        logger.error(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()