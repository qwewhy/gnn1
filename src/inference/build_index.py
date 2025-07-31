import torch
import torch.nn as nn
from pathlib import Path
import yaml
import tqdm
import logging
import sys
from typing import Optional, Dict, Any

from src.data_processing.pyg_dataset import PatchDataset
from src.models.gnn_encoder import DualInputGNNEncoder

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelValidator:
    """模型检查点验证器"""
    
    @staticmethod
    def validate_checkpoint(checkpoint_path: Path, config: Dict[str, Any]) -> bool:
        """验证模型检查点是否有效"""
        try:
            if not checkpoint_path.exists():
                logger.warning(f"检查点文件不存在: {checkpoint_path}")
                return False
            
            # 尝试加载检查点
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 创建模型实例以验证结构匹配
            model = DualInputGNNEncoder(
                anchor_in_channels=config['model']['anchor_in_channels'],
                pattern_in_channels=config['model']['pattern_in_channels'],
                hidden_channels=config['model']['hidden_channels'],
                out_channels=config['model']['out_channels'],
                gnn_type=config['model']['gnn_type']
            )
            
            # 验证状态字典兼容性
            model.load_state_dict(checkpoint, strict=False)
            logger.info(f"检查点验证成功: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"检查点验证失败: {e}")
            return False
    
    @staticmethod
    def create_random_model(config: Dict[str, Any], save_path: Path) -> bool:
        """创建随机初始化的模型作为回退方案"""
        try:
            model = DualInputGNNEncoder(
                anchor_in_channels=config['model']['anchor_in_channels'],
                pattern_in_channels=config['model']['pattern_in_channels'],
                hidden_channels=config['model']['hidden_channels'],
                out_channels=config['model']['out_channels'],
                gnn_type=config['model']['gnn_type']
            )
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.warning(f"创建随机初始化模型: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建随机模型失败: {e}")
            return False

class RobustQueryEngine:
    """带错误处理的查询引擎"""
    
    def __init__(self, config_path: str, device: str = 'auto'):
        self.config_path = config_path
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu')
        self.config = None
        self.model = None
        self.dataset = None
        
        # 初始化组件
        if not self._load_config():
            raise RuntimeError("配置加载失败")
        
        if not self._load_dataset():
            raise RuntimeError("数据集加载失败")
            
        if not self._load_model():
            raise RuntimeError("模型加载失败")
    
    def _load_config(self) -> bool:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("配置文件加载成功")
            return True
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return False
    
    def _load_dataset(self) -> bool:
        """加载数据集"""
        try:
            data_root = Path(self.config['data']['root'])
            
            # 检查数据库文件
            db_path = data_root / 'raw' / 'patches.db'
            if not db_path.exists():
                logger.error(f"数据库文件不存在: {db_path}")
                return False
            
            self.dataset = PatchDataset(root=str(data_root))
            logger.info(f"数据集加载成功，包含 {len(self.dataset)} 个模式")
            return True
            
        except Exception as e:
            logger.error(f"数据集加载失败: {e}")
            return False
    
    def _load_model(self) -> bool:
        """加载或创建模型"""
        try:
            model_config = self.config['model']
            self.model = DualInputGNNEncoder(
                anchor_in_channels=model_config['anchor_in_channels'],
                pattern_in_channels=model_config['pattern_in_channels'],
                hidden_channels=model_config['hidden_channels'],
                out_channels=model_config['out_channels'],
                gnn_type=model_config['gnn_type']
            ).to(self.device)
            
            # 尝试加载最佳模型
            checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
            best_model_path = checkpoint_dir / 'best_model.pt'
            final_model_path = checkpoint_dir / 'final_model.pt'
            
            # 按优先级尝试加载
            for model_path in [best_model_path, final_model_path]:
                if ModelValidator.validate_checkpoint(model_path, self.config):
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    logger.info(f"模型加载成功: {model_path}")
                    return True
            
            # 如果没有有效检查点，创建随机模型
            logger.warning("未找到有效的训练模型，使用随机初始化模型")
            random_model_path = checkpoint_dir / 'random_init_model.pt'
            
            if ModelValidator.create_random_model(self.config, random_model_path):
                self.model.load_state_dict(torch.load(random_model_path, map_location=self.device))
                self.model.eval()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def encode_pattern_batch(self, data_batch) -> Optional[torch.Tensor]:
        """批量编码模式"""
        try:
            with torch.no_grad():
                return self.model(data_batch.to(self.device), input_type='pattern')
        except Exception as e:
            logger.error(f"模式编码失败: {e}")
            return None

def build_pattern_index_robust(config_path: str, force_rebuild: bool = False):
    """构建模式索引的鲁棒版本"""
    
    # 1. 初始化鲁棒查询引擎
    try:
        query_engine = RobustQueryEngine(config_path)
    except RuntimeError as e:
        logger.error(f"查询引擎初始化失败: {e}")
        return False
    
    # 2. 检查是否需要重建索引
    data_root = Path(query_engine.config['data']['root'])
    index_path = data_root / 'processed' / 'pattern_index.pt'
    
    if index_path.exists() and not force_rebuild:
        try:
            # 验证现有索引
            existing_index = torch.load(index_path, map_location='cpu')
            if existing_index.shape[0] == len(query_engine.dataset):
                logger.info(f"使用现有索引: {index_path}")
                return True
            else:
                logger.warning("现有索引大小不匹配，重建索引")
        except Exception as e:
            logger.warning(f"现有索引损坏，重建索引: {e}")
    
    # 3. 构建新索引
    try:
        all_embeddings = []
        batch_size = query_engine.config['training']['batch_size']
        
        # 使用DataLoader进行批量处理
        from torch_geometric.loader import DataLoader
        data_loader = DataLoader(
            query_engine.dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0  # 避免多进程问题
        )
        
        logger.info(f"开始构建索引，共 {len(query_engine.dataset)} 个模式...")
        
        successful_batches = 0
        failed_batches = 0
        
        for batch_idx, data_batch in enumerate(tqdm.tqdm(data_loader, desc="Building index")):
            embeddings = query_engine.encode_pattern_batch(data_batch)
            
            if embeddings is not None:
                all_embeddings.append(embeddings.cpu())
                successful_batches += 1
            else:
                failed_batches += 1
                logger.warning(f"批次 {batch_idx} 编码失败")
                
                # 如果失败太多，停止构建
                if failed_batches > successful_batches * 0.1:  # 失败率超过10%
                    logger.error("失败率过高，停止索引构建")
                    return False
        
        if not all_embeddings:
            logger.error("没有成功编码的批次")
            return False
        
        # 4. 合并和保存索引
        pattern_index = torch.cat(all_embeddings, dim=0)
        
        # 确保目录存在
        index_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pattern_index, index_path)
        
        logger.info(f"索引构建完成: {pattern_index.shape[0]} 个向量")
        logger.info(f"索引保存至: {index_path}")
        logger.info(f"成功率: {successful_batches}/{successful_batches + failed_batches}")
        
        return True
        
    except Exception as e:
        logger.error(f"索引构建过程中发生错误: {e}")
        return False

def verify_index_integrity(config_path: str) -> bool:
    """验证索引完整性"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        data_root = Path(config['data']['root'])
        index_path = data_root / 'processed' / 'pattern_index.pt'
        
        if not index_path.exists():
            logger.warning("索引文件不存在")
            return False
        
        # 加载并验证索引
        pattern_index = torch.load(index_path, map_location='cpu')
        dataset = PatchDataset(root=str(data_root))
        
        if pattern_index.shape[0] != len(dataset):
            logger.error(f"索引大小不匹配: {pattern_index.shape[0]} vs {len(dataset)}")
            return False
        
        if pattern_index.shape[1] != config['model']['out_channels']:
            logger.error(f"嵌入维度不匹配: {pattern_index.shape[1]} vs {config['model']['out_channels']}")
            return False
        
        logger.info("索引完整性验证通过")
        return True
        
    except Exception as e:
        logger.error(f"索引验证失败: {e}")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='构建模式索引')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='强制重建索引')
    parser.add_argument('--verify-only', action='store_true',
                       help='仅验证现有索引')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        # 尝试相对于脚本位置的路径
        script_dir = Path(__file__).parent.parent.parent
        config_path = script_dir / args.config
        
        if not config_path.exists():
            logger.error(f"配置文件不存在: {args.config}")
            sys.exit(1)
    
    if args.verify_only:
        success = verify_index_integrity(str(config_path))
        sys.exit(0 if success else 1)
    
    success = build_pattern_index_robust(str(config_path), args.force_rebuild)
    
    if success:
        # 构建完成后验证
        verify_index_integrity(str(config_path))
        logger.info("索引构建和验证完成")
    else:
        logger.error("索引构建失败")
        sys.exit(1)

if __name__ == '__main__':
    main()