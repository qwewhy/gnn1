#!/usr/bin/env python3
# File: run_api.py
# API启动脚本

"""
启动FastAPI后端服务
使用这个脚本替代直接运行main.py，提供更好的错误处理和日志
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """设置环境变量和路径"""
    # 解决OpenMP问题
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # 设置输出编码
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # 确保项目根目录在Python路径中
    project_root = Path(__file__).parent.absolute()
    sys.path.insert(0, str(project_root))
    
    try:
        print(f"📁 项目根目录: {project_root}")
    except UnicodeEncodeError:
        print(f"Project root: {project_root}")
    return project_root

def check_dependencies():
    """检查必要的依赖"""
    required_modules = [
        'fastapi', 'uvicorn', 'pydantic', 
        'torch', 'numpy', 'trimesh', 'plotly', 'yaml'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        try:
            print(f"❌ 缺少以下依赖: {missing_modules}")
            print("💡 请运行: pip install -r src/use/api/requirements.txt")
        except UnicodeEncodeError:
            print(f"Missing dependencies: {missing_modules}")
            print("Please run: pip install -r src/use/api/requirements.txt")
        return False
    
    try:
        print("✅ 所有依赖检查通过")
    except UnicodeEncodeError:
        print("All dependencies check passed")
    return True

def main():
    """主函数"""
    # Windows兼容的输出
    try:
        print("🚀 启动3D面片可视化查询API服务")
    except UnicodeEncodeError:
        print("Starting 3D Mesh Patch Visual Query API Service")
    
    print("=" * 60)
    
    # 设置环境
    project_root = setup_environment()
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 检查关键文件
    api_main = project_root / 'src' / 'use' / 'api' / 'main.py'
    if not api_main.exists():
        try:
            print(f"❌ API主文件不存在: {api_main}")
        except UnicodeEncodeError:
            print(f"API main file not found: {api_main}")
        return 1
    
    config_file = project_root / 'configs' / 'config.yaml'
    if not config_file.exists():
        try:
            print(f"❌ 配置文件不存在: {config_file}")
            print("💡 请确保配置文件存在于 configs/config.yaml")
        except UnicodeEncodeError:
            print(f"Config file not found: {config_file}")
            print("Please ensure config file exists at configs/config.yaml")
        return 1
    
    try:
        # 导入并启动API
        try:
            print("📦 导入API模块...")
        except UnicodeEncodeError:
            print("Importing API module...")
        from src.use.api.main import app
        
        try:
            print("🌐 启动FastAPI服务器...")
            print("📍 API地址: http://localhost:8000")
            print("📖 API文档: http://localhost:8000/docs")
            print("🔍 健康检查: http://localhost:8000/api/health")
            print("⏹️  按 Ctrl+C 停止服务")
        except UnicodeEncodeError:
            print("Starting FastAPI server...")
            print("API address: http://localhost:8000")
            print("API docs: http://localhost:8000/docs")
            print("Health check: http://localhost:8000/api/health")
            print("Press Ctrl+C to stop service")
        print("-" * 60)
        
        import uvicorn
        uvicorn.run(
            "src.use.api.main:app", 
            host="0.0.0.0", 
            port=8000, 
            reload=True,
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        try:
            print("\n👋 服务已停止")
        except UnicodeEncodeError:
            print("\nService stopped")
        return 0
    except Exception as e:
        try:
            print(f"❌ 启动失败: {e}")
        except UnicodeEncodeError:
            print(f"Startup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


