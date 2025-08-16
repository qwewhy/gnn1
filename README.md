# 1
$env:KMP_DUPLICATE_LIB_OK="TRUE"
在控制台提前允许包重复，排除错误

# 2
python -m src.train.training.improved_train

# 3
# 运行集成训练示例
python src/train/training/integrated_training_example.py

# 4
基础训练模式（快速验证）
bashpython src/train/training/improved_train.py --config configs/config.yaml

# 5
硬挖掘增强模式（提升效果）
修改配置文件，设置 use_hard_mining: true：
bashpython src/train/training/improved_train.py --config configs/hard_mining_config.yaml

# 6
高级多任务模式（完整功能）
bashpython src/train/training/advanced_training_system.py

# 7
from src.train.training.integrated_training_example import IntegratedTrainingSystem

# 对比不同训练方法
system = IntegratedTrainingSystem('configs/config.yaml')
results = system.compare_training_systems()

# 查看对比结果
for method, result in results.items():
    print(f"{method}: {result['status']}")

# 查看挖掘统计
stats = miner.get_mining_statistics()
print(f"挖掘成功率: {stats['success_rate']:.2f}")
print(f"回退率: {stats['fallback_rate']:.2f}")
print(f"平均三元组数: {stats['average_triplets_per_batch']:.1f}")



