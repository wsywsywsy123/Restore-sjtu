# 🏛️ 壁画病害诊断系统

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

**方法1: 使用启动脚本（推荐）**
```bash
python start_app.py
```

**方法2: 直接启动**
```bash
python -m streamlit run app.py --server.port 8501
```

**方法3: 使用批处理文件（Windows）**
双击运行 `run_app.bat` 或 `启动应用.bat`

4. 访问应用
启动后，在浏览器中打开以下任一地址：
- http://localhost:8501
- http://127.0.0.1:8501

### 故障排除

如果应用无法打开，请按以下步骤检查：

1. **检查应用是否运行**
   ```bash
   python check_app.py
   ```
   或运行诊断工具：
   ```bash
   python diagnose.py
   ```

2. **常见问题解决**

   **问题1: 端口被占用**
   - 错误信息: `Port 8501 is already in use`
   - 解决方法: 运行 `python start_app.py` 会自动处理端口占用
   - 或手动终止进程后重启

   **问题2: 浏览器无法连接**
   - 等待10-30秒让应用完全启动
   - 尝试使用 `http://127.0.0.1:8501` 代替 `localhost`
   - 检查Windows防火墙设置
   - 检查杀毒软件是否阻止连接

   **问题3: 依赖包缺失**
   - 运行: `pip install -r requirements.txt`
   - 确保Python版本 >= 3.8

   **问题4: 路径编码问题（Windows中文路径）**
   - 使用 `start_app.py` 启动（Python脚本，避免编码问题）
   - 或使用英文路径重命名项目文件夹

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

## 工具说明

- `start_app.py` - 智能启动脚本（自动处理端口占用）
- `check_app.py` - 检查应用运行状态
- `diagnose.py` - 诊断工具（检查连接和常见问题）
- `run_app.bat` - Windows批处理启动脚本
- `启动应用.bat` - Windows批处理启动脚本（自动处理端口）

## 联系方式

- 项目地址: [GitHub Repository]
- 问题反馈: [Issues]
