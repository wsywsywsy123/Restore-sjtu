# start_backend.py - 启动后端API服务
import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """安装后端依赖"""
    requirements = [
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "opencv-python-headless",
        "numpy",
        "pillow",
        "scikit-learn",
        "joblib"
    ]
    
    print("正在安装后端依赖...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"{req} 安装成功")
        except subprocess.CalledProcessError:
            print(f"{req} 安装失败")

def start_backend():
    """启动后端服务"""
    print("启动壁画病害诊断后端API服务...")
    print("服务地址: http://localhost:8000")
    print("API文档: http://localhost:8000/docs")
    print("按 Ctrl+C 停止服务")
    
    try:
        # 首先尝试启动完整版API
        try:
            subprocess.run([sys.executable, "backend_api.py"])
        except Exception as e:
            print(f"完整版API启动失败: {e}")
            print("尝试启动简化版API...")
            subprocess.run([sys.executable, "simple_backend_api.py"])
    except KeyboardInterrupt:
        print("\n服务已停止")

if __name__ == "__main__":
    # 检查是否需要安装依赖
    try:
        import fastapi
        import uvicorn
    except ImportError:
        install_requirements()
    
    start_backend()
