# File: src/use/build_index.py
# 模式索引构建模块 / Pattern index building module

import torch
import yaml
from pathlib import Path
import sys
import argparse
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 使用统一的路径管理器
try:
    from src.common.path_manager import setup_project_environment
except ImportError:
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / 'src'))
    from src.common.path_manager import setup_project_environment

path_manager = setup_project_environment()

# 导入项目模块
from src.train.models.improved_gat_encoder import MetricLearningGAT
from src.train.data_processing.pyg_dataset import PatchDataset
from torch_geometric.loader import DataLoader

def build_index(config_path: str):
    """
    使用训练好的模型为数据集中的所有模式生成嵌入向量，并保存为索引文件。
    """
    # 1. 加载配置
    logger.info(f"正在加载配置文件: {config_path}")
    config_file = path_manager.get_config_path(Path(config_path).name)
    if not config_file.exists():
        logger.error(f"配置文件不存在: {config_file}")
        return

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 3. 加载数据集
    logger.info("正在加载数据集...")
    try:
        dataset = PatchDataset(root=str(path_manager.data_dir))
        loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)
        logger.info(f"数据集加载成功，包含 {len(dataset)} 个模式。")
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        logger.error("请确保数据库 'data/raw/patches.db' 已生成，并且PyG数据集文件存在或可以被创建。")
        return

    # 4. 加载模型
    logger.info("正在加载训练好的模型...")
    model_config = config['model']
    model = MetricLearningGAT(
        anchor_in_channels=model_config['anchor_in_channels'],
        pattern_in_channels=model_config['pattern_in_channels'],
        hidden_channels=model_config['hidden_channels'],
        out_channels=model_config['out_channels'],
        num_heads=model_config['num_heads'],
        edge_dim=model_config['edge_dim']
    ).to(device)

    checkpoint_path = path_manager.project_root / config['training']['checkpoint_dir'] / 'best_model.pt'
    if not checkpoint_path.exists():
        logger.error(f"找不到模型检查点: {checkpoint_path}")
        logger.error("请先运行训练脚本。")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("模型加载成功。")

    # 5. 生成嵌入向量
    logger.info("正在为所有模式生成嵌入向量...")
    all_embeddings = []
    all_pattern_ids = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # 注意：这里我们只编码 "anchor" 分支，因为所有数据都是独立的模式
            embeddings = model.encoder_pattern(data)
            all_embeddings.append(embeddings.cpu())
            all_pattern_ids.extend([d.item() for d in data.pattern_id])

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    logger.info(f"嵌入向量生成完毕，形状为: {embeddings_tensor.shape}")

    # 6. 保存索引
    index_data = {
        'embeddings': embeddings_tensor,
        'pattern_ids': all_pattern_ids
    }
    
    index_save_path = path_manager.data_processed_dir / 'pattern_index.pt'
    path_manager.ensure_directory_exists(index_save_path.parent) # 确保目录存在
    
    torch.save(index_data, index_save_path)
    logger.info(f"模式索引已成功保存到: {index_save_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='为数据集构建模式索引')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件的名称 (例如 config.yaml)')
    args = parser.parse_args()

    build_index(config_path=args.config)

if __name__ == '__main__':
    main()
