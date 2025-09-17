## 石窟寺壁画病害 AI 识别与复原工具（Streamlit）

功能概览：
- 病害检测（裂缝、剥落、褪色、污渍/霉斑、盐蚀/风化、生物附着）
- 自动材质识别（启发式/可选 ONNX 模型）并联动评分建议
- 细化病理指标（连通域/面积/形态/方向）与 CSV 导出
- PDF 报告导出（含标注图与建议）
- 试验性图像复原（Inpainting：Telea/Navier）

### 1) 本地运行

1. Python 3.13（或兼容版本），安装依赖：
```
pip install -r requirements.txt
pip install streamlit opencv-python-headless onnxruntime pillow pandas reportlab
```
2. 运行：
```
streamlit run app.py
```

如需使用 ONNX 分割或材质模型，在侧栏填入模型路径。

### 2) 部署到 Streamlit Community Cloud

1. 将本仓库推送到 GitHub 公有仓库
2. 在 `streamlit.io` 选择 New app → 选中仓库与分支 → 入口文件 `app.py`
3. 首次部署会自动安装 `requirements.txt` 并生成公网链接

### 3) 部署到 Hugging Face Spaces（Streamlit）

1. 新建 Space，SDK 选 Streamlit
2. 将代码与 `requirements.txt` 上传或指向 GitHub 仓库
3. Spaces 会自动安装依赖并生成链接

### 4) Docker 自托管（可选）

```
docker build -t mural-ai .
docker run -p 8501:8501 mural-ai
```
然后访问 `http://localhost:8501`

### 5) 采集扩展数据（可选）

使用 `collect_images.py` 采集病理类别图片并生成清单：
```
python collect_images.py --per_category 100
```

### 许可与数据
- 学术研究用途优先。采集到的网络图片请遵守各站点版权与使用条款。


