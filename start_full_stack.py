#!/usr/bin/env python3
# File: start_full_stack.py
# 一键启动完整的前后端系统

"""
一键启动脚本 - 同时启动FastAPI后端和React前端
使用此脚本可以方便地启动整个全栈应用
"""

import os
import sys
import subprocess
import time
import signal
import platform
from pathlib import Path
import threading
import webbrowser

# 全局进程列表，用于清理
running_processes = []

def setup_environment():
    """设置环境变量"""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    print("✅ 环境变量设置完成")

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查系统依赖...")
    
    # 检查Python
    try:
        import uvicorn
        import fastapi
        print("✅ Python和FastAPI依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少Python依赖: {e}")
        print("💡 请运行: pip install -r src/use/api/requirements.txt")
        return False
    
    # 检查Node.js和npm
    try:
        result = subprocess.run(['node', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Node.js: {result.stdout.strip()}")
        else:
            raise subprocess.CalledProcessError(result.returncode, 'node')
            
        # Windows下特殊处理npm检查
        if platform.system() == "Windows":
            # 在Windows上尝试多种方式检查npm
            npm_commands = [
                ['npm', '--version'],
                ['powershell', '-Command', 'npm --version'],
                ['cmd', '/c', 'npm --version'],
                ['npx', '--version']  # 备用方案
            ]
            
            npm_found = False
            for npm_cmd in npm_commands:
                try:
                    result = subprocess.run(npm_cmd, 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        if version and not 'error' in version.lower():
                            print(f"✅ npm: {version}")
                            npm_found = True
                            break
                except Exception:
                    continue
            
            if not npm_found:
                print("⚠️ npm检查失败，但将尝试继续（可能是PowerShell执行策略问题）")
                print("💡 如果遇到问题，请以管理员身份运行 PowerShell 并执行:")
                print("   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser")
                # 不返回False，继续尝试
        else:
            # 非Windows系统的标准检查
            result = subprocess.run(['npm', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ npm: {result.stdout.strip()}")
            else:
                raise subprocess.CalledProcessError(result.returncode, 'npm')
            
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ Node.js或npm未安装或不在PATH中")
        print("💡 请安装Node.js: https://nodejs.org/")
        return False
    
    return True

def install_frontend_dependencies():
    """安装前端依赖"""
    frontend_dir = Path(__file__).parent / 'frontend'
    package_json = frontend_dir / 'package.json'
    node_modules = frontend_dir / 'node_modules'
    
    if not package_json.exists():
        print("❌ 未找到frontend/package.json")
        return False
    
    if not node_modules.exists():
        print("📦 安装前端依赖...")
        try:
            # Windows下尝试多种npm执行方式
            if platform.system() == "Windows":
                npm_commands = [
                    ['cmd', '/c', 'npm install'],
                    ['powershell', '-Command', 'npm install'],
                    ['npm', 'install']
                ]
            else:
                npm_commands = [['npm', 'install']]
            
            success = False
            for npm_cmd in npm_commands:
                try:
                    print(f"💡 尝试命令: {' '.join(npm_cmd)}")
                    result = subprocess.run(
                        npm_cmd, 
                        cwd=frontend_dir, 
                        timeout=300,  # 5分钟超时
                        capture_output=True,
                        text=True,
                        shell=(platform.system() == "Windows" and npm_cmd[0] not in ['cmd', 'powershell'])
                    )
                    if result.returncode == 0:
                        print("✅ 前端依赖安装成功")
                        success = True
                        break
                    else:
                        print(f"⚠️ 命令失败: {result.stderr[:200]}...")
                        continue
                except subprocess.TimeoutExpired:
                    print("⚠️ 命令超时，尝试下一个...")
                    continue
                except Exception as e:
                    print(f"⚠️ 命令出错: {e}")
                    continue
            
            if not success:
                print("❌ 所有npm install命令都失败了")
                print("💡 请手动运行: cd frontend && npm install")
                return False
            
            return True
        except Exception as e:
            print(f"❌ 前端依赖安装出错: {e}")
            return False
    else:
        print("✅ 前端依赖已存在")
        return True

def start_backend():
    """启动后端服务"""
    print("🚀 启动FastAPI后端服务...")
    
    project_root = Path(__file__).parent
    api_script = project_root / 'run_api.py'
    
    if not api_script.exists():
        print(f"❌ 后端启动脚本不存在: {api_script}")
        return None
    
    try:
        # 设置环境变量确保正确的编码
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        env['OMP_NUM_THREADS'] = '1'
        
        # 使用当前Python解释器启动后端
        process = subprocess.Popen(
            [sys.executable, str(api_script)],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
            encoding='utf-8',
            errors='ignore'  # 忽略编码错误而不是失败
        )
        
        running_processes.append(process)
        
        # 启动线程来处理输出
        def handle_backend_output():
            try:
                for line in process.stdout:
                    if line.strip():  # 只输出非空行
                        try:
                            # 在Windows上更好地处理编码
                            decoded_line = line.strip()
                            if decoded_line:
                                print(f"[Backend] {decoded_line}")
                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                            # 跳过有编码问题的行，不打印错误
                            pass
            except Exception as e:
                # 静默处理输出错误，避免干扰主要功能
                pass
        
        output_thread = threading.Thread(target=handle_backend_output, daemon=True)
        output_thread.start()
        
        # 等待服务启动并验证
        print("⏳ 等待后端服务启动...")
        time.sleep(8)  # 给更多时间让后端完成初始化
        
        if process.poll() is None:
            # 验证API是否真正可用
            try:
                import requests
                
                # 等待并测试健康检查端点
                for i in range(10):  # 最多等待20秒
                    try:
                        response = requests.get('http://localhost:8000/api/health', timeout=2)
                        if response.status_code == 200:
                            health_data = response.json()
                            if health_data.get('components_initialized', False):
                                print("✅ 后端服务启动成功，所有组件已初始化 (http://localhost:8000)")
                                return process
                            else:
                                print(f"⏳ 后端正在初始化组件... ({i+1}/10)")
                                time.sleep(2)
                                continue
                        else:
                            print(f"⏳ 等待后端响应... ({i+1}/10)")
                            time.sleep(2)
                    except requests.exceptions.RequestException:
                        print(f"⏳ 等待后端启动... ({i+1}/10)")
                        time.sleep(2)
                
                print("⚠️ 后端服务已启动但组件初始化可能未完成")
                return process
                
            except ImportError:
                print("✅ 后端服务启动成功 (http://localhost:8000) - 无法验证状态(缺少requests库)")
                return process
        else:
            print("❌ 后端服务启动失败")
            return None
            
    except Exception as e:
        print(f"❌ 启动后端服务出错: {e}")
        return None

def start_frontend():
    """启动前端服务"""
    print("🎨 启动React前端服务...")
    
    frontend_dir = Path(__file__).parent / 'frontend'
    
    if not frontend_dir.exists():
        print(f"❌ 前端目录不存在: {frontend_dir}")
        return None
    
    try:
        # 设置环境变量以避免自动打开浏览器
        env = os.environ.copy()
        env['BROWSER'] = 'none'
        
        # Windows下使用合适的命令启动npm
        if platform.system() == "Windows":
            # 在Windows上使用cmd来启动npm start
            start_command = ['cmd', '/c', 'npm start']
            shell = False
        else:
            start_command = ['npm', 'start']
            shell = False
        
        print(f"💡 启动命令: {' '.join(start_command)}")
        process = subprocess.Popen(
            start_command,
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
            shell=shell
        )
        
        running_processes.append(process)
        
        # 启动线程来处理输出
        def handle_frontend_output():
            try:
                for line in process.stdout:
                    if line.strip():  # 只输出非空行
                        try:
                            # 在Windows上更好地处理编码
                            decoded_line = line.strip()
                            if decoded_line:
                                print(f"[Frontend] {decoded_line}")
                        except (UnicodeDecodeError, UnicodeEncodeError) as e:
                            # 跳过有编码问题的行，不打印错误
                            pass
            except Exception as e:
                # 静默处理输出错误，避免干扰主要功能
                pass
        
        output_thread = threading.Thread(target=handle_frontend_output, daemon=True)
        output_thread.start()
        
        # 等待服务启动
        print("⏳ 等待前端服务启动...")
        time.sleep(10)  # React启动需要更长时间
        
        if process.poll() is None:
            print("✅ 前端服务启动成功 (http://localhost:3000)")
            return process
        else:
            print("❌ 前端服务启动失败")
            return None
            
    except Exception as e:
        print(f"❌ 启动前端服务出错: {e}")
        return None

def open_browser():
    """打开浏览器"""
    try:
        time.sleep(2)  # 等待服务完全启动
        print("🌐 正在打开浏览器...")
        webbrowser.open('http://localhost:3000')
        print("✅ 浏览器已打开")
    except Exception as e:
        print(f"⚠️ 无法自动打开浏览器: {e}")
        print("💡 请手动访问: http://localhost:3000")

def cleanup_processes():
    """清理所有进程"""
    print("\n🧹 正在清理进程...")
    
    for process in running_processes:
        try:
            if process.poll() is None:  # 进程仍在运行
                if platform.system() == "Windows":
                    process.terminate()
                else:
                    process.send_signal(signal.SIGTERM)
                
                # 等待进程结束
                try:
                    process.wait(timeout=5)
                    print(f"✅ 进程 {process.pid} 已正常终止")
                except subprocess.TimeoutExpired:
                    print(f"⚠️ 进程 {process.pid} 未响应，强制终止")
                    process.kill()
                    process.wait()
        except Exception as e:
            print(f"⚠️ 清理进程时出错: {e}")

def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n🛑 收到信号 {signum}，正在停止服务...")
    cleanup_processes()
    print("👋 服务已停止")
    sys.exit(0)

def main():
    """主函数"""
    print("🎯 启动3D网格面片可视化查询系统")
    print("=" * 60)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    if platform.system() != "Windows":
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 设置环境
        setup_environment()
        
        # 检查依赖
        if not check_dependencies():
            print("❌ 依赖检查失败，无法启动")
            return 1
        
        # 安装前端依赖
        if not install_frontend_dependencies():
            print("❌ 前端依赖安装失败，无法启动")
            return 1
        
        # 启动后端
        backend_process = start_backend()
        if not backend_process:
            print("❌ 后端启动失败，无法继续")
            return 1
        
        # 启动前端
        frontend_process = start_frontend()
        if not frontend_process:
            print("❌ 前端启动失败")
            cleanup_processes()
            return 1
        
        # 打开浏览器
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # 打印成功信息
        print("\n" + "=" * 60)
        print("🎉 全栈应用启动成功！")
        print("📍 后端API: http://localhost:8000")
        print("🌐 前端界面: http://localhost:3000")
        print("📖 API文档: http://localhost:8000/docs")
        print("🔍 健康检查: http://localhost:8000/api/health")
        print("⏹️  按 Ctrl+C 停止所有服务")
        print("=" * 60)
        
        # 等待进程结束
        try:
            while True:
                # 检查进程状态
                backend_alive = backend_process.poll() is None
                frontend_alive = frontend_process.poll() is None
                
                if not backend_alive:
                    print("⚠️ 后端服务已停止")
                    break
                if not frontend_alive:
                    print("⚠️ 前端服务已停止")
                    break
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n🛑 收到中断信号")
        
        cleanup_processes()
        return 0
        
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        cleanup_processes()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


