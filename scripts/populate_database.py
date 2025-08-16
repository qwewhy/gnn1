#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root)) # 应该添加项目根目录，以便src可以作为包被找到

from src.train.data_processing.populate_db import main

if __name__ == "__main__":
    print("🚀 Running Database Population Script...")
    main()
    print("✅ Database Population Finished.")
