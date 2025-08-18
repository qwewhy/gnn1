# File: run_map_evaluation.py
# 专门的mAP检索评估脚本 / Dedicated mAP retrieval evaluation script

# 修复OMP错误 - 必须在其他库导入之前设置
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import torch
import logging
import argparse

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# 导入项目模块
from src.train.data_processing.pyg_dataset import PatchDataset
from src.common.path_manager import setup_project_environment
from src.train.models.improved_gat_encoder import ImprovedGATEncoder
from src.train.evaluation.retrieval_metrics import RetrievalEvaluator


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='mAP检索性能评估工具')
    
    parser.add_argument('--model_path', type=str, default='src/train/training/checkpoints/advanced/best_model.pt',
                       help='训练好的模型权重路径')
    parser.add_argument('--config_path', type=str, default='configs/advanced_training_config.yaml',
                       help='模型配置文件路径')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='最大评估样本数（0表示全部）')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='评估批次大小')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10, 20],
                       help='计算Precision@K和Recall@K的K值列表')
    parser.add_argument('--analyze_failures', action='store_true',
                       help='是否分析检索失败案例')
    parser.add_argument('--save_embeddings', action='store_true',
                       help='是否保存嵌入向量')
    parser.add_argument('--output_dir', type=str, default='output/retrieval_evaluation',
                       help='输出目录')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    
    return parser.parse_args()


def setup_logging(verbose=False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_model_and_dataset(model_path, config_path, logger):
    """加载模型和数据集"""
    # 设置项目环境
    path_manager = setup_project_environment()
    dataset_path = path_manager.data_processed_dir
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")
    
    # 加载数据集
    dataset = PatchDataset(root=str(dataset_path))
    logger.info(f"✅ 数据集加载成功: {len(dataset)} 个样本")
    
    if len(dataset) == 0:
        raise ValueError("数据集为空")
    
    # 检查模型文件
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建模型
    sample_data = dataset[0]
    
    # 从配置文件或样本数据推断模型配置
    try:
        import yaml
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            model_config = config.get('model', {})
        else:
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            model_config = {}
    except Exception as e:
        logger.warning(f"无法加载配置文件: {e}，使用默认配置")
        model_config = {}
    
    # 设置模型参数
    encoder_config = {
        'anchor_in_channels': model_config.get('anchor_in_channels', 8),
        'pattern_in_channels': model_config.get('pattern_in_channels', 8),
        'hidden_channels': model_config.get('hidden_channels', 64),
        'out_channels': model_config.get('out_channels', 128),
        'num_heads': model_config.get('num_heads', 4),
        'edge_dim': model_config.get('edge_dim', 3),
        'dropout': 0.0  # 评估时不使用dropout
    }
    
    model = ImprovedGATEncoder(**encoder_config)
    
    # 加载训练好的权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        
        # 检查是否是AdvancedGATModel的状态（包含base_encoder前缀）
        if any(key.startswith('base_encoder.') for key in state_dict.keys()):
            logger.info("检测到AdvancedGATModel格式，提取base_encoder部分")
            # 提取base_encoder的权重
            encoder_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('base_encoder.'):
                    new_key = key.replace('base_encoder.', '')
                    encoder_state_dict[new_key] = value
            model.load_state_dict(encoder_state_dict)
        else:
            # 直接加载ImprovedGATEncoder格式
            model.load_state_dict(state_dict)
        
        model.eval()
        logger.info(f"✅ 模型加载成功: {model_path}")
    except Exception as e:
        raise RuntimeError(f"无法加载模型权重: {e}")
    
    logger.info(f"📱 使用设备: {device}")
    
    return model, dataset, device


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置日志
    logger = setup_logging(args.verbose)
    
    print("🎯 mAP检索性能评估工具")
    print("=" * 60)
    
    try:
        # 加载模型和数据集
        logger.info("🔄 加载模型和数据集...")
        model, dataset, device = load_model_and_dataset(args.model_path, args.config_path, logger)
        
        # 创建检索评估器
        evaluator = RetrievalEvaluator(k_values=args.k_values)
        
        # 限制样本数
        max_samples = args.max_samples if args.max_samples > 0 else None
        
        # 评估检索性能
        logger.info("🎯 开始检索性能评估...")
        retrieval_results = evaluator.evaluate_model_retrieval(
            model=model,
            dataset=dataset,
            device=device,
            max_samples=max_samples,
            batch_size=args.batch_size,
            save_embeddings=args.save_embeddings,
            output_dir=args.output_dir if args.save_embeddings else None
        )
        
        # 打印详细结果
        evaluator.print_detailed_results(retrieval_results)
        
        # 分析失败案例
        if args.analyze_failures:
            logger.info("🔍 分析检索失败案例...")
            failure_cases, success_cases = evaluator.analyze_failure_cases(
                model, dataset, device, num_examples=10, k=5
            )
            
            print(f"\n📊 案例分析总结:")
            print(f"   失败案例数: {len(failure_cases)}")
            print(f"   成功案例数: {len(success_cases)}")
        
        # 保存结果
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存评估结果
        import json
        results_file = output_dir / 'retrieval_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(retrieval_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 结果已保存到: {results_file}")
        
        # 性能总结
        map_score = retrieval_results.get('mAP', 0)
        print(f"\n🏆 性能总结:")
        print(f"   mAP: {map_score:.4f}")
        print(f"   Precision@5: {retrieval_results.get('precision@5', 0):.4f}")
        print(f"   Precision@10: {retrieval_results.get('precision@10', 0):.4f}")
        
        if map_score > 0.7:
            print("   🎉 检索性能优秀！")
        elif map_score > 0.5:
            print("   ✅ 检索性能良好")
        elif map_score > 0.3:
            print("   ⚠️ 检索性能一般，需要改进")
        else:
            print("   ❌ 检索性能较差，需要大幅改进")
        
        print("\n✅ 评估完成！")
        
    except FileNotFoundError as e:
        logger.error(f"❌ 文件不存在: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"❌ 数据错误: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"❌ 运行时错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
