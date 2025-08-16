#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.train.training.improved_train import main

if __name__ == "__main__":
    print("🚀 Running Training Script...")
    main()
    print("✅ Training Finished.")
