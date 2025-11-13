# deploy_config.py - 部署配置
import os
from pathlib import Path

class DeployConfig:
    """部署配置类"""
    
    # 部署选项
    DEPLOY_OPTIONS = {
        "streamlit_cloud": {
            "name": "Streamlit Community Cloud",
            "description": "免费的Streamlit托管服务",
            "url": "https://share.streamlit.io",
            "requirements": ["streamlit", "opencv-python-headless", "numpy", "pandas", "pillow", "requests"],
            "setup_commands": [
                "pip install -r requirements.txt",
                "streamlit run app.py"
            ]
        },
        "huggingface_spaces": {
            "name": "Hugging Face Spaces",
            "description": "免费的机器学习模型托管平台",
            "url": "https://huggingface.co/spaces",
            "requirements": ["streamlit", "opencv-python-headless", "numpy", "pandas", "pillow", "requests"],
            "setup_commands": [
                "pip install -r requirements.txt",
                "streamlit run app.py"
            ]
        },
        "railway": {
            "name": "Railway",
            "description": "现代化的云部署平台",
            "url": "https://railway.app",
            "requirements": ["streamlit", "opencv-python-headless", "numpy", "pandas", "pillow", "requests"],
            "setup_commands": [
                "pip install -r requirements.txt",
                "streamlit run app.py --server.port $PORT"
            ]
        },
        "render": {
            "name": "Render",
            "description": "全栈应用托管平台",
            "url": "https://render.com",
            "requirements": ["streamlit", "opencv-python-headless", "numpy", "pandas", "pillow", "requests"],
            "setup_commands": [
                "pip install -r requirements.txt",
                "streamlit run app.py --server.port $PORT"
            ]
        }
    }
    
    @staticmethod
    def generate_requirements():
        """生成requirements.txt文件"""
        requirements = [
            "streamlit>=1.28.0",
            "opencv-python-headless>=4.8.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "pillow>=10.0.0",
            "requests>=2.31.0",
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "python-multipart>=0.0.6",
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0",
            "sqlite3",
            "pathlib"
        ]
        
        with open("requirements.txt", "w", encoding="utf-8") as f:
            for req in requirements:
                f.write(f"{req}\n")
        
        print("requirements.txt 已生成")
    
    @staticmethod
    def generate_streamlit_config():
        """生成Streamlit配置文件"""
        config_content = """
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#8B0000"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
        
        with open(".streamlit/config.toml", "w", encoding="utf-8") as f:
            f.write(config_content)
        
        print("Streamlit配置已生成")
    
    @staticmethod
    def generate_dockerfile():
        """生成Dockerfile"""
        dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p persistent_data/images

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""
        
        with open("Dockerfile", "w", encoding="utf-8") as f:
            f.write(dockerfile_content)
        
        print("Dockerfile 已生成")
    
    @staticmethod
    def generate_github_workflow():
        """生成GitHub Actions工作流"""
        workflow_content = """name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Test application
      run: |
        python -c "import streamlit; print('Streamlit imported successfully')"
        python -c "import cv2; print('OpenCV imported successfully')"
        python -c "import numpy; print('NumPy imported successfully')"
"""
        
        os.makedirs(".github/workflows", exist_ok=True)
        with open(".github/workflows/deploy.yml", "w", encoding="utf-8") as f:
            f.write(workflow_content)
        
        print("GitHub Actions工作流已生成")
    
    @staticmethod
    def generate_readme():
        """生成README.md"""
        readme_content = """# 🏛️ 壁画病害诊断系统

## 项目简介

这是一个基于AI的壁画病害诊断系统，支持：
- 二维壁画病害识别
- 三维石窟监测
- 文献资料OCR识别
- 多模态融合诊断
- 壁画数据库管理

## 功能特点

- 🤖 **AI智能诊断**: 使用机器学习算法识别壁画病害
- 📊 **数据管理**: 支持图片上传、分类、标注
- 🔄 **数据持久化**: 数据永久保存，不会丢失
- 🌐 **云端部署**: 支持多种云平台部署
- 📱 **响应式设计**: 支持各种设备访问

## 快速开始

### 本地运行

1. 克隆项目
```bash
git clone <your-repo-url>
cd Restore
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 启动应用
```bash
python start_app.py
```

4. 访问应用
打开浏览器访问: http://localhost:8501

### 云端部署

#### Streamlit Community Cloud
1. 将代码推送到GitHub
2. 访问 https://share.streamlit.io
3. 连接GitHub仓库
4. 选择主分支和app.py文件
5. 点击Deploy

#### Hugging Face Spaces
1. 在Hugging Face创建新的Space
2. 选择Streamlit SDK
3. 上传代码文件
4. 等待自动部署

## 数据管理

系统使用SQLite数据库进行数据持久化存储：
- 图片数据永久保存
- 支持数据导出和备份
- 多用户数据隔离

## 技术栈

- **前端**: Streamlit
- **后端**: FastAPI
- **数据库**: SQLite
- **图像处理**: OpenCV
- **机器学习**: scikit-learn
- **部署**: Docker, Streamlit Cloud

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 联系方式

- 项目地址: [GitHub Repository]
- 问题反馈: [Issues]
"""
        
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        print("README.md 已生成")

if __name__ == "__main__":
    print("生成部署配置文件...")
    
    # 创建.streamlit目录
    os.makedirs(".streamlit", exist_ok=True)
    
    # 生成各种配置文件
    DeployConfig.generate_requirements()
    DeployConfig.generate_streamlit_config()
    DeployConfig.generate_dockerfile()
    DeployConfig.generate_github_workflow()
    DeployConfig.generate_readme()
    
    print("\n所有部署配置文件已生成完成！")
    print("\n部署选项:")
    for key, config in DeployConfig.DEPLOY_OPTIONS.items():
        print(f"  {key}: {config['name']} - {config['description']}")
