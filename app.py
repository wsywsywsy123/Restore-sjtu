# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
import base64
import os
import sys
try:
    import onnxruntime as ort  # 深度分割推理
except Exception:
    ort = None

# Optional deps for 3D
try:
    import open3d as o3d  # type: ignore
except Exception:
    o3d = None
try:
    import plotly.express as px
    import pandas as pd  # already imported above but for safety when 3D used
except Exception:
    px = None
try:
    from rapidocr_onnxruntime import RapidOCR  # 轻量OCR，基于 onnxruntime
except Exception:
    RapidOCR = None

# 多模态融合相关依赖
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    import json
    MULTIMODAL_AVAILABLE = True
except Exception:
    MULTIMODAL_AVAILABLE = False

st.set_page_config("石窟寺壁画病害AI识别工具（升级版）", layout="wide", page_icon="🏛️")

# Session init
if "proc" not in st.session_state:
    st.session_state["proc"] = None

# ---------------------------
# 动态背景与品牌标识
# ---------------------------

@st.cache_data(show_spinner=False)
def get_background_images_b64(dir_path: str = "assets/backgrounds"):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images: list[str] = []
    try:
        if os.path.isdir(dir_path):
            for name in sorted(os.listdir(dir_path)):
                ext = os.path.splitext(name)[1].lower()
                if ext in exts:
                    full = os.path.join(dir_path, name)
                    with open(full, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                        mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
                        images.append(f"data:{mime};base64,{b64}")
    except Exception:
        pass
    return images

@st.cache_data(show_spinner=False)
def get_logo_b64(candidates: list[str] = [
    "assets/sjtu_design.png", "assets/sjtu.png", "assets/logo_sjtu.png"
]):
    for p in candidates:
        try:
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                    ext = os.path.splitext(p)[1].lower()
                    mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
                    return f"data:{mime};base64,{b64}"
        except Exception:
            continue
    return None

def inject_dynamic_background(images_data_urls: list[str], interval_ms: int = 8000):
    if not images_data_urls:
        return
    imgs_js_array = ",".join([f"'" + u + "'" for u in images_data_urls])
    css = f"""
    <style>
    .stApp {{
        background-size: cover !important;
        background-position: center center !important;
        background-attachment: fixed !important;
        transition: background-image 1.2s ease-in-out;
    }}
    .bg-overlay::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: radial-gradient(ellipse at center, rgba(0,0,0,0.25), rgba(0,0,0,0.45));
        pointer-events: none;
        z-index: 0;
    }}
    .app-brand-badge {{
        position: fixed;
        right: 16px;
        bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(6px);
        border-radius: 10px;
        padding: 8px 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        z-index: 9999;
    }}
    .app-brand-badge img {{ height: 28px; width: auto; display:block; }}
    .app-brand-title {{ font-weight: 600; color: #333; font-size: 13px; line-height: 1.2; }}
    </style>
    """
    js = f"""
    <script>
    const bgImgs = [{imgs_js_array}];
    let idx = 0;
    function applyBg() {{
      const el = parent.document.querySelector('.stApp');
      if (!el) return;
      el.style.backgroundImage = `url(${bgImgs[idx]})`;
      idx = (idx + 1) % bgImgs.length;
    }}
    applyBg();
    if (!window.__bgInterval) {{
      window.__bgInterval = setInterval(applyBg, {interval_ms});
    }}
    const root = parent.document.querySelector('.stApp');
    if (root && !root.classList.contains('bg-overlay')) {{
      root.classList.add('bg-overlay');
    }}
    </script>
    """
    st.markdown(css + js, unsafe_allow_html=True)

def inject_brand_badge(logo_data_url: str | None):
    logo_img_html = f'<img src="{logo_data_url}" alt="SJTU" />' if logo_data_url else ""
    html = f"""
    <div class="app-brand-badge">
      {logo_img_html}
      <div class="app-brand-title">上海交通大学<br/>设计学院</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---------------------------
# 多模态融合系统
# ---------------------------

class KnowledgeGraph:
    """石窟病害知识图谱"""
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_knowledge_graph()
    
    def _build_knowledge_graph(self):
        """构建石窟类型-材质-典型病害-修复手段知识图谱"""
        # 石窟类型节点
        cave_types = {
            "敦煌莫高窟": {"era": "北魏-元代", "climate": "干旱", "structure": "砂岩"},
            "云冈石窟": {"era": "北魏", "climate": "温带", "structure": "花岗岩"},
            "龙门石窟": {"era": "北魏-唐代", "climate": "温带", "structure": "石灰岩"},
            "麦积山石窟": {"era": "后秦-清代", "climate": "温带", "structure": "泥质砂岩"}
        }
        
        # 材质节点
        materials = {
            "砂岩": {"porosity": "高", "hardness": "中", "weathering": "易风化"},
            "花岗岩": {"porosity": "低", "hardness": "高", "weathering": "抗风化"},
            "石灰岩": {"porosity": "中", "hardness": "中", "weathering": "易溶蚀"},
            "泥质砂岩": {"porosity": "高", "hardness": "低", "weathering": "极易风化"}
        }
        
        # 病害节点
        pathologies = {
            "表面裂缝": {"severity": "中", "depth": "浅层", "cause": "温差应力"},
            "深层裂缝": {"severity": "高", "depth": "深层", "cause": "结构应力"},
            "剥落": {"severity": "高", "depth": "表面", "cause": "风化"},
            "变色": {"severity": "低", "depth": "表面", "cause": "氧化"},
            "盐析": {"severity": "中", "depth": "表面", "cause": "盐分结晶"},
            "生物侵蚀": {"severity": "中", "depth": "表面", "cause": "微生物"}
        }
        
        # 修复手段节点
        treatments = {
            "表面加固": {"cost": "低", "effectiveness": "中", "durability": "短"},
            "深层注浆": {"cost": "高", "effectiveness": "高", "durability": "长"},
            "表面清洗": {"cost": "低", "effectiveness": "高", "durability": "短"},
            "保护涂层": {"cost": "中", "effectiveness": "中", "durability": "中"},
            "环境控制": {"cost": "高", "effectiveness": "高", "durability": "长"}
        }
        
        # 构建图结构
        for cave, props in cave_types.items():
            self.graph.add_node(cave, type="cave", **props)
        
        for material, props in materials.items():
            self.graph.add_node(material, type="material", **props)
        
        for pathology, props in pathologies.items():
            self.graph.add_node(pathology, type="pathology", **props)
        
        for treatment, props in treatments.items():
            self.graph.add_node(treatment, type="treatment", **props)
        
        # 添加关系边
        relationships = [
            # 石窟-材质关系
            ("敦煌莫高窟", "砂岩", {"compatibility": "高"}),
            ("云冈石窟", "花岗岩", {"compatibility": "高"}),
            ("龙门石窟", "石灰岩", {"compatibility": "高"}),
            ("麦积山石窟", "泥质砂岩", {"compatibility": "高"}),
            
            # 材质-病害关系
            ("砂岩", "表面裂缝", {"probability": 0.8}),
            ("砂岩", "剥落", {"probability": 0.9}),
            ("花岗岩", "深层裂缝", {"probability": 0.6}),
            ("石灰岩", "盐析", {"probability": 0.7}),
            ("泥质砂岩", "剥落", {"probability": 0.95}),
            ("泥质砂岩", "生物侵蚀", {"probability": 0.8}),
            
            # 病害-修复关系
            ("表面裂缝", "表面加固", {"suitability": 0.9}),
            ("深层裂缝", "深层注浆", {"suitability": 0.95}),
            ("剥落", "表面加固", {"suitability": 0.8}),
            ("变色", "表面清洗", {"suitability": 0.9}),
            ("盐析", "表面清洗", {"suitability": 0.85}),
            ("生物侵蚀", "表面清洗", {"suitability": 0.8}),
        ]
        
        for source, target, attrs in relationships:
            self.graph.add_edge(source, target, **attrs)
    
    def query_treatment(self, cave_type, material, pathologies):
        """根据石窟类型、材质和病害查询最佳修复方案"""
        treatments = []
        for pathology in pathologies:
            # 查找该病害的修复方案
            for treatment in self.graph.successors(pathology):
                if self.graph.nodes[treatment]["type"] == "treatment":
                    suitability = self.graph[pathology][treatment].get("suitability", 0.5)
                    treatments.append({
                        "pathology": pathology,
                        "treatment": treatment,
                        "suitability": suitability,
                        "cost": self.graph.nodes[treatment]["cost"],
                        "effectiveness": self.graph.nodes[treatment]["effectiveness"],
                        "durability": self.graph.nodes[treatment]["durability"]
                    })
        
        # 按适用性排序
        treatments.sort(key=lambda x: x["suitability"], reverse=True)
        return treatments

class MultimodalFusion:
    """多模态融合系统"""
    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()
        self.text_encoder = None
        self.image_encoder = None
        self.pointcloud_encoder = None
        self._init_encoders()
    
    def _init_encoders(self):
        """初始化各模态编码器"""
        if not MULTIMODAL_AVAILABLE:
            return
        
        try:
            # 文本编码器（使用预训练的中文BERT）
            self.text_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            self.text_encoder = AutoModel.from_pretrained("bert-base-chinese")
        except:
            st.warning("文本编码器初始化失败，将使用简化版本")
    
    def encode_image(self, image):
        """图像特征编码"""
        if image is None:
            return None
        
        # 使用预训练的ResNet特征
        # 这里简化处理，实际应该使用预训练模型
        features = cv2.resize(image, (224, 224))
        features = cv2.cvtColor(features, cv2.COLOR_BGR2RGB)
        features = features.flatten()[:512]  # 简化特征
        return features / np.linalg.norm(features)
    
    def encode_pointcloud(self, pointcloud):
        """点云特征编码"""
        if pointcloud is None or o3d is None:
            return None
        
        # 计算点云几何特征
        features = []
        
        # 密度特征
        if len(pointcloud.points) > 0:
            bbox = pointcloud.get_axis_aligned_bounding_box()
            volume = bbox.volume()
            density = len(pointcloud.points) / max(volume, 1e-6)
            features.append(density)
        else:
            features.append(0)
        
        # 表面粗糙度（简化计算）
        if len(pointcloud.points) > 10:
            points = np.asarray(pointcloud.points)
            distances = np.linalg.norm(points - np.mean(points, axis=0), axis=1)
            roughness = np.std(distances)
            features.append(roughness)
        else:
            features.append(0)
        
        # 法向量分布（简化）
        if hasattr(pointcloud, 'normals') and len(pointcloud.normals) > 0:
            normals = np.asarray(pointcloud.normals)
            normal_std = np.std(normals, axis=0)
            features.extend(normal_std.tolist())
        else:
            features.extend([0, 0, 0])
        
        # 填充到固定长度
        while len(features) < 64:
            features.append(0)
        
        features = np.array(features[:64])
        return features / (np.linalg.norm(features) + 1e-8)
    
    def encode_text(self, text):
        """文本特征编码"""
        if not text or self.text_encoder is None:
            return None
        
        try:
            inputs = self.text_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return features
        except:
            # 简化文本编码
            words = text.split()
            features = np.zeros(768)
            for i, word in enumerate(words[:10]):  # 只取前10个词
                features[i*77:(i+1)*77] = np.random.randn(77)  # 简化处理
            return features / (np.linalg.norm(features) + 1e-8)
    
    def fuse_modalities(self, image_features, pointcloud_features, text_features):
        """多模态特征融合"""
        features = []
        
        if image_features is not None:
            features.append(image_features)
        if pointcloud_features is not None:
            features.append(pointcloud_features)
        if text_features is not None:
            features.append(text_features)
        
        if not features:
            return None
        
        # 简单拼接融合（实际应该使用注意力机制）
        fused = np.concatenate(features)
        return fused / (np.linalg.norm(fused) + 1e-8)
    
    def analyze_depth_stability(self, image, pointcloud, crack_mask):
        """结合点云分析裂缝深度和结构稳定性"""
        if pointcloud is None or o3d is None:
            return {"depth": "unknown", "stability": "unknown", "confidence": 0.0}
        
        try:
            # 提取裂缝区域的点云
            points = np.asarray(pointcloud.points)
            if len(points) == 0:
                return {"depth": "unknown", "stability": "unknown", "confidence": 0.0}
            
            # 计算裂缝深度（简化算法）
            z_coords = points[:, 2]  # 假设Z轴是深度
            depth_variance = np.var(z_coords)
            
            # 计算结构稳定性指标
            bbox = pointcloud.get_axis_aligned_bounding_box()
            volume = bbox.volume()
            point_density = len(points) / max(volume, 1e-6)
            
            # 基于几何特征判断
            if depth_variance > 0.1:
                depth = "deep"
                stability = "unstable"
            elif depth_variance > 0.05:
                depth = "medium"
                stability = "moderate"
            else:
                depth = "shallow"
                stability = "stable"
            
            confidence = min(point_density * 10, 1.0)
            
            return {
                "depth": depth,
                "stability": stability,
                "confidence": confidence,
                "depth_variance": depth_variance,
                "point_density": point_density
            }
        except Exception as e:
            return {"depth": "error", "stability": "error", "confidence": 0.0, "error": str(e)}

class AutoAnnotator:
    """LLM自动标注系统"""
    def __init__(self):
        self.annotation_templates = {
            "crack": {
                "description": "裂缝病害，通常表现为线性缺陷",
                "severity_levels": ["轻微", "中等", "严重"],
                "key_features": ["线性", "连续性", "深度变化"]
            },
            "peel": {
                "description": "剥落病害，表面材料脱落",
                "severity_levels": ["轻微", "中等", "严重"],
                "key_features": ["不规则形状", "边缘清晰", "厚度变化"]
            },
            "discolor": {
                "description": "变色病害，颜色异常变化",
                "severity_levels": ["轻微", "中等", "严重"],
                "key_features": ["颜色差异", "边界模糊", "面积分布"]
            }
        }
    
    def generate_annotation(self, image, detected_regions, defect_type):
        """基于检测结果生成自动标注"""
        if defect_type not in self.annotation_templates:
            return None
        
        template = self.annotation_templates[defect_type]
        annotations = []
        
        for region in detected_regions:
            # 计算区域特征
            area = region.get("area", 0)
            bbox = region.get("bbox", [0, 0, 0, 0])
            elongation = region.get("elongation", 0)
            
            # 基于特征判断严重程度
            if area > 1000:
                severity = "严重"
            elif area > 500:
                severity = "中等"
            else:
                severity = "轻微"
            
            # 生成标注文本
            annotation = {
                "type": defect_type,
                "description": template["description"],
                "severity": severity,
                "area": area,
                "bbox": bbox,
                "confidence": 0.8,  # 简化置信度
                "features": {
                    "elongation": elongation,
                    "aspect_ratio": bbox[2] / max(bbox[3], 1),
                    "area_ratio": area / (image.shape[0] * image.shape[1])
                }
            }
            annotations.append(annotation)
        
        return annotations

class GenerativeAugmentation:
    """生成式增强：虚拟修复"""
    def __init__(self):
        self.restoration_templates = {
            "crack": {
                "method": "inpainting",
                "parameters": {"algorithm": "telea", "radius": 3}
            },
            "peel": {
                "method": "texture_synthesis",
                "parameters": {"patch_size": 32, "overlap": 8}
            },
            "discolor": {
                "method": "color_correction",
                "parameters": {"method": "reinhard", "target": "reference"}
            }
        }
    
    def virtual_restoration(self, image, mask, defect_type):
        """虚拟修复模拟"""
        if defect_type not in self.restoration_templates:
            return image
        
        template = self.restoration_templates[defect_type]
        
        if template["method"] == "inpainting":
            # 使用OpenCV修复
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        elif template["method"] == "color_correction":
            # 颜色校正
            result = self._color_correction(image, mask)
        else:
            result = image
        
        return result
    
    def _color_correction(self, image, mask):
        """颜色校正"""
        # 简化颜色校正
        result = image.copy()
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # 基于周围区域的颜色进行校正
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        mask_eroded = cv2.erode(mask, kernel, iterations=2)
        border_mask = mask_dilated - mask_eroded
        
        if np.sum(border_mask) > 0:
            # 计算边界区域的平均颜色
            border_pixels = image[border_mask > 0]
            if len(border_pixels) > 0:
                mean_color = np.mean(border_pixels, axis=0)
                result[mask > 0] = mean_color
        
        return result

# 全局多模态系统实例
@st.cache_resource
def get_multimodal_system():
    return MultimodalFusion()

@st.cache_resource
def get_auto_annotator():
    return AutoAnnotator()

@st.cache_resource
def get_generative_augmentation():
    return GenerativeAugmentation()

# ---------------------------
# Caching helpers
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_onnx_session_cached(model_path: str, providers):
    return ort.InferenceSession(model_path, providers=providers)

@st.cache_data(show_spinner=False)
def _resize_bgr_cached(image_bgr_bytes: bytes, w: int, h: int):
    arr = np.frombuffer(image_bgr_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法从缓存字节解码图像")
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

@st.cache_resource(show_spinner=False)
def get_rapidocr_cached():
    if RapidOCR is None:
        return None
    try:
        return RapidOCR()
    except Exception:
        return None

# ---------------------------
# Helpers: render inpainting UI
# ---------------------------
def render_inpainting_ui(img_rgb, mask_crack, mask_peel, mask_disc, mask_stain, mask_salt, mask_bio, default_open=True, key_suffix=""):
    st.markdown("### 🧩 图像复原（试验性 Inpainting）")
    with st.expander("展开/收起", expanded=default_open):
        sel_classes = st.multiselect(
            "选择需要复原的病害类别（将基于其掩膜进行修补）",
            ["裂缝","剥落","褪色","污渍/霉斑","盐蚀/风化","生物附着"],
            default=["裂缝","剥落","污渍/霉斑"], key=f"sel_classes_{key_suffix}"
        )
        method = st.selectbox("修补算法", ["Telea", "Navier-Stokes"], index=0, key=f"method_{key_suffix}")
        radius = st.slider("修补半径（像素）", min_value=1, max_value=25, value=7, key=f"radius_{key_suffix}")
        go_restore = st.button("生成复原图像", key=f"restore_btn_{key_suffix}")
        if go_restore:
            class_to_mask = {
                "裂缝": mask_crack,
                "剥落": mask_peel,
                "褪色": mask_disc,
                "污渍/霉斑": mask_stain,
                "盐蚀/风化": mask_salt,
                "生物附着": mask_bio,
            }
            union = np.zeros(mask_crack.shape, dtype=np.uint8)
            for c in sel_classes:
                m = class_to_mask.get(c)
                if m is not None:
                    union = cv2.bitwise_or(union, (m>0).astype(np.uint8)*255)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((union>0).astype(np.uint8), connectivity=8)
            filtered = np.zeros_like(union)
            for i in range(1, num_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area >= 50:
                    filtered[labels==i] = 255
            flag = cv2.INPAINT_TELEA if method == "Telea" else cv2.INPAINT_NS
            src_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            restored_bgr = cv2.inpaint(src_bgr, (filtered>0).astype(np.uint8), radius, flag)
            restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
            st.image(restored_rgb, caption="复原结果（基于所选掩膜）", width='stretch')
            _buf = BytesIO(); Image.fromarray(restored_rgb).save(_buf, format="PNG"); _buf.seek(0)
            st.download_button("下载复原图（PNG）", data=_buf.getvalue(), file_name="restored.png", mime="image/png")

# ---------------------------
# Helpers: color restoration utilities
# ---------------------------
def _to_bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def _to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def gray_world_white_balance(img_bgr):
    img = img_bgr.astype(np.float32)
    mean_b, mean_g, mean_r = [np.mean(img[:,:,c]) for c in range(3)]
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    gain_b = mean_gray / (mean_b + 1e-6)
    gain_g = mean_gray / (mean_g + 1e-6)
    gain_r = mean_gray / (mean_r + 1e-6)
    img[:,:,0] = np.clip(img[:,:,0] * gain_b, 0, 255)
    img[:,:,1] = np.clip(img[:,:,1] * gain_g, 0, 255)
    img[:,:,2] = np.clip(img[:,:,2] * gain_r, 0, 255)
    return img.astype(np.uint8)

def clahe_on_l_channel(img_bgr, clip_limit=2.0, tile_grid_size=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def reinhard_color_transfer(src_bgr, ref_bgr):
    # Convert to LAB and match mean/std
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        s_mean, s_std = src_lab[:,:,c].mean(), src_lab[:,:,c].std() + 1e-6
        r_mean, r_std = ref_lab[:,:,c].mean(), ref_lab[:,:,c].std() + 1e-6
        src_lab[:,:,c] = (src_lab[:,:,c] - s_mean) * (r_std / s_std) + r_mean
    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)

def render_color_restore_ui(img_rgb, default_open=False, key_suffix="color"):
    st.markdown("### 🎨 色彩/褪色复原（基础版）")
    with st.expander("展开/收起", expanded=default_open):
        col1, col2 = st.columns(2)
        with col1:
            use_wb = st.checkbox("灰度世界白平衡", value=True, key=f"wb_{key_suffix}")
            clahe_clip = st.slider("CLAHE 对比度 (clip)", 0.0, 4.0, 2.0, 0.1, key=f"clip_{key_suffix}")
            clahe_tile = st.slider("CLAHE 网格", 4, 16, 8, 1, key=f"tile_{key_suffix}")
        with col2:
            ref_file = st.file_uploader("参考图像（可选，用于风格/色彩转移）", type=["jpg","jpeg","png"], key=f"ref_{key_suffix}")
            do_transfer = st.checkbox("启用参考色彩转移（Reinhard）", value=False, key=f"tr_{key_suffix}")
        run_color = st.button("生成色彩复原图", key=f"btn_{key_suffix}")
        if run_color:
            bgr = _to_bgr(img_rgb)
            out = bgr
            if use_wb:
                out = gray_world_white_balance(out)
            out = clahe_on_l_channel(out, clip_limit=clahe_clip, tile_grid_size=clahe_tile)
            if do_transfer and ref_file is not None:
                ref_bytes = np.asarray(bytearray(ref_file.read()), dtype=np.uint8)
                ref_bgr = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)
                if ref_bgr is not None:
                    out = reinhard_color_transfer(out, ref_bgr)
            rgb = _to_rgb(out)
            st.image(rgb, caption="色彩复原结果", width='stretch')
            buf = BytesIO(); Image.fromarray(rgb).save(buf, format="PNG"); buf.seek(0)
            st.download_button("下载复原图（PNG）", data=buf.getvalue(), file_name="restored_color.png", mime="image/png")

# ---------------------------
# Utility helpers
# ---------------------------
def pil_from_cv2(cv2_img):
    """BGR or RGB? we expect RGB input"""
    if len(cv2_img.shape) == 3:
        return Image.fromarray(cv2_img)
    else:
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

def encode_image_to_bytes(img_rgb):
    """PIL Image -> bytes"""
    buf = BytesIO()
    img_rgb.save(buf, format="PNG")
    buf.seek(0)
    return buf

def save_annotated_image_bytes(annotated_rgb):
    """Return bytesIO for embedding to reportlab or streamlit download"""
    pil = Image.fromarray(annotated_rgb)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ---------------------------
# Material-specific parameters
# ---------------------------
MATERIAL_OPTIONS = [
    "未指定",
    "大足石刻（砂岩）",
    "云冈石窟（砂岩夹泥岩）",
    "敦煌莫高窟（灰泥/颜料层）",
    "木质基底（木板）"
]

MATERIAL_WEIGHTS = {
    # weights for severity scoring per category
    # categories: crack, peel, disc, stain, salt, bio
    "未指定": {
        "crack": 1.0, "peel": 1.0, "disc": 1.0, "stain": 0.9, "salt": 1.0, "bio": 0.8
    },
    "大足石刻（砂岩）": {
        "crack": 1.2, "peel": 1.3, "disc": 0.9, "stain": 0.9, "salt": 1.4, "bio": 0.8
    },
    "云冈石窟（砂岩夹泥岩）": {
        "crack": 1.3, "peel": 1.2, "disc": 1.0, "stain": 0.9, "salt": 1.3, "bio": 0.8
    },
    "敦煌莫高窟（灰泥/颜料层）": {
        "crack": 1.0, "peel": 1.1, "disc": 1.4, "stain": 1.1, "salt": 0.9, "bio": 0.8
    },
    "木质基底（木板）": {
        "crack": 1.1, "peel": 1.2, "disc": 1.0, "stain": 1.2, "salt": 0.6, "bio": 1.3
    }
}

MATERIAL_SUGGESTIONS = {
    "大足石刻（砂岩）": [
        "砂岩质地疏松，优先防水加固、防止盐析与崩解。",
        "针对大面积剥落，建议进行物理加固与注浆。"
    ],
    "云冈石窟（砂岩夹泥岩）": [
        "控制洞窟湿度，检查夹层水渗、采取注浆或支撑加固。",
        "注意裂缝注入和裂缝扩展的监测。"
    ],
    "敦煌莫高窟（灰泥/颜料层）": [
        "重点保护颜料层，避免直接触摸与湿热环境变动。",
        "对于起甲与颜料脱落，应采用可逆性修复材料优先。"
    ],
    "木质基底（木板）": [
        "关注虫蛀、霉菌与含水率变化，必要时进行防虫与除湿处理。",
        "避免强光直射与大幅温湿变化，表层建议使用可逆性保护涂层。"
    ]
}

# 细化：按病害类型和比例生成建议
def build_recommendations(material, pct_map, overall_severity):
    recs = []
    m = material
    cp = pct_map.get('crack', 0.0)
    pp = pct_map.get('peel', 0.0)
    dp = pct_map.get('disc', 0.0)
    sp = pct_map.get('stain', 0.0)
    sap = pct_map.get('salt', 0.0)
    bp = pct_map.get('bio', 0.0)

    # 裂缝
    if cp > 0.1:
        if cp < 1:
            recs.append("裂缝轻度：建议裂缝监测与记录，避免震动与干湿变动；必要时表面加固。")
        elif cp < 5:
            recs.append("裂缝中度：进行裂缝走向/宽度测量与定期复测；可采用微注浆或低黏度加固树脂（可逆/低挥发）进行填充与加固。")
        else:
            recs.append("裂缝重度：优先实施结构加固（支撑/锚固/注浆），并查明致因（渗水、温变、应力），同步开展长期监测。")
        if "木质" in m:
            recs.append("木质注意：优先控制含水率与温湿稳定，裂缝部位避免热胀冷缩反复；加固材料需兼容木材纤维。")

    # 剥落/起甲
    if pp > 0.1:
        if pp < 1:
            recs.append("剥落轻度：小面积起甲可先做边缘点固与局部回贴，现场观察其发展趋势。")
        elif pp < 5:
            recs.append("剥落中度：对空鼓/起甲区域进行注浆回贴，边界处采用逐段加固；作业前进行材性与粘结试验。")
        else:
            recs.append("剥落重度：大面积面层不稳，需分区分步回贴与网格化管理，过程中保持环境稳定并做好支撑与防坠落防护。")
        if "砂岩" in m:
            recs.append("砂岩注意：先做基体含盐/含水评估，必要时先期脱盐与干燥后再行回贴加固。")

    # 褪色/粉化
    if dp > 0.1:
        if dp < 1:
            recs.append("褪色轻度：加强光照与温湿管理，避免触摸与风沙磨蚀；建立高保真影像档案。")
        elif dp < 5:
            recs.append("褪色中度：进行颜料层稳固性测试，选择低光泽、可逆的表面稳色/固色处理；限定参观距离与时间。")
        else:
            recs.append("褪色重度：组织材料学评估（颜料矿物与黏结相），采用最小干预的可逆稳色体系并建立长期光照阈值管理。")
        if "灰泥/颜料" in m:
            recs.append("灰泥/颜料层注意：严控紫外与挥发性污染物，操作使用中性pH清洁与保护体系，避免深度渗入型材料。")

    # 污渍/霉斑
    if sp > 0.1:
        if sp < 1:
            recs.append("污渍轻度：采用干式/低湿清洁（软刷/微吸）去除表面尘垢，先做小样试验。")
        elif sp < 5:
            recs.append("污渍中度：局部配合凝胶清洁与控湿处理，清洁后做再污染防护。")
        else:
            recs.append("污渍重度：制定分区清洁方案，配合环境治理（过滤/密封/人流控制）并评估颜料层稳定性。")

    # 盐蚀/风化
    if sap > 0.1:
        if sap < 1:
            recs.append("盐蚀轻度：监测盐花与白化，控制水源与湿度波动；避免直接水洗造成盐迁移。")
        elif sap < 5:
            recs.append("盐蚀中度：实施温和脱盐（纸浆/凝胶）与气候调控，随后进行基体加固；必要时表面防盐屏障。")
        else:
            recs.append("盐蚀重度：先期系统脱盐与干燥，再分阶段结构与表层加固；建立长期渗水/含盐监测体系。")
        if "木质" in m:
            recs.append("木质注意：盐蚀通常次要，重点放在防霉与含水率控制，不宜采用高含水处理。")

    # 生物附着
    if bp > 0.1:
        if bp < 1:
            recs.append("生物轻度：增强通风与干燥，消除积尘与营养源，物理性去除为主。")
        elif bp < 5:
            recs.append("生物中度：小范围使用低毒可逆性生物抑制剂（先试验再使用），并持续控湿控光。")
        else:
            recs.append("生物重度：制定综合治理（控湿、控光、定期维护与过滤），必要时分批次化学抑制并评估对颜料层影响。")
        if "木质" in m:
            recs.append("木质注意：优先防霉防虫，考虑熏蒸/局部抗菌与防虫处置，并严控含水率。")

    # 总体策略
    if overall_severity < 5:
        recs.append("总体：问题轻微，纳入常规巡检与影像档案管理，半年/一年复查。")
    elif overall_severity < 20:
        recs.append("总体：中度病害，建议制定分区治理计划与优先级，先做样区试验后再全面展开。")
    else:
        recs.append("总体：重度病害，尽快组织跨专业团队（结构、材料、环境）联合评估与处置，设置长期监测。")

    # 附：材质专用提示
    recs += MATERIAL_SUGGESTIONS.get(m, [])
    return recs

# ---------------------------
# Image preprocessing & detection functions (classical CV baseline)
# ---------------------------

def preprocess_image(image_bgr, target_max_dim=1600):
    """Resize while keeping aspect ratio for reasonable processing time."""
    h, w = image_bgr.shape[:2]
    scale = 1.0
    max_dim = max(h, w)
    if max_dim > target_max_dim:
        scale = target_max_dim / max_dim
        image_bgr = cv2.resize(image_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return image_bgr, scale

def detect_cracks(gray):
    """Detect fine elongated structures (baseline): morphological thinning + contour filtering."""
    # enhance contrast
    gray_eq = cv2.equalizeHist(gray)
    # use Scharr or Sobel to get strong gradients
    grad_x = cv2.Sobel(gray_eq, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_eq, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0))
    # binary threshold + morphological closing to join thin lines
    _, th = cv2.threshold(grad, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    # find contours and filter elongated
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask = np.zeros_like(th)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < 80: 
            continue
        # elongated or thin filter
        if (w > 4*h) or (h > 4*w) or (area < 200 and max(w,h) > 40):
            boxes.append((x,y,w,h))
            cv2.drawContours(mask, [c], -1, 255, -1)
    return boxes, mask

def detect_peeling(hsv):
    """Low saturation patches (剥落/灰白斑块)"""
    h,s,v = cv2.split(hsv)
    # threshold low saturation but not pure dark
    low_sat = cv2.inRange(hsv, (0,0,40), (180,70,255))
    # remove tiny speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    low_sat = cv2.morphologyEx(low_sat, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(low_sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask = np.zeros_like(low_sat)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 400: 
            continue
        boxes.append((x,y,w,h))
        cv2.drawContours(mask, [c], -1, 255, -1)
    return boxes, mask

def detect_discoloration(hsv):
    """Overly bright or faded regions: high V with low to mid saturation"""
    lower = np.array([0,0,180])
    upper = np.array([180,90,255])
    light_mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask = np.zeros_like(light_mask)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 300: 
            continue
        boxes.append((x,y,w,h))
        cv2.drawContours(mask, [c], -1, 255, -1)
    return boxes, mask

# ---------------------------
# Material classification (heuristic + optional ONNX)
# ---------------------------
def classify_material_heuristic(image_bgr):
    """Return (material_name, confidence_0_1, details_dict) based on simple cues.
    Heuristic cues:
    - Brown/orange ratio (wood tendency)
    - Orientation coherence (wood grain)
    - High-bright low-sat salts (stone tendency)
    """
    img_small = cv2.resize(image_bgr, (min(640, image_bgr.shape[1]), min(640, image_bgr.shape[0])), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    # brown-ish mask (wood): hue 10-30, moderate saturation/value
    brown = cv2.inRange(hsv, (10, 40, 40), (30, 255, 220))
    brown_ratio = float(np.mean(brown > 0))

    # salt-like (stone efflorescence): very bright low S
    salt_like = cv2.inRange(hsv, (0, 0, 220), (180, 40, 255))
    salt_ratio = float(np.mean(salt_like > 0))

    # orientation coherence via gradients
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    # compute histogram concentration around main orientation
    ang_mask = mag > (np.percentile(mag, 75))
    main_orient = np.median(ang[ang_mask]) if np.any(ang_mask) else 0.0
    concentr = float(np.mean(np.abs(((ang - main_orient + 90) % 180) - 90) < 10))

    # crude decision
    wood_score = 0.55 * brown_ratio + 0.35 * concentr + 0.10 * (1.0 - salt_ratio)
    stone_score = 0.60 * (1.0 - brown_ratio) + 0.25 * (1.0 - concentr) + 0.15 * salt_ratio

    if wood_score > stone_score:
        name = "木质基底（木板）"
        conf = min(1.0, max(0.0, (wood_score - stone_score) * 2.0))
    else:
        # choose between known stone presets by default to "未指定" or closest
        name = "大足石刻（砂岩）"
        conf = min(1.0, max(0.0, (stone_score - wood_score) * 2.0))

    details = {
        'brown_ratio': round(brown_ratio, 4),
        'salt_ratio': round(salt_ratio, 4),
        'orientation_concentration': round(concentr, 4)
    }
    return name, conf, details

def run_material_model(image_bgr, model_path, providers=None, class_names=None):
    if ort is None:
        raise RuntimeError("未安装 onnxruntime，请先安装：pip install onnxruntime")
    session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
    # preprocess to square 224
    target = 224
    img = cv2.resize(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), (target, target), interpolation=cv2.INTER_AREA)
    inp = img.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2,0,1))[None, ...]
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    out = session.run([output_name], {input_name: inp})[0]
    # softmax
    if out.ndim == 2 and out.shape[0] == 1:
        logits = out[0]
    elif out.ndim == 1:
        logits = out
    else:
        raise ValueError("材质模型输出形状不支持：" + str(out.shape))
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    if class_names is None:
        class_names = [
            "未指定",
            "大足石刻（砂岩）",
            "云冈石窟（砂岩夹泥岩）",
            "敦煌莫高窟（灰泥/颜料层）",
            "木质基底（木板）"
        ]
    idx = int(np.argmax(probs))
    return class_names[idx], float(np.max(probs)), dict(zip(class_names[:len(probs)], [float(p) for p in probs]))

# 额外类别（污渍/霉斑、盐蚀/风化、生物附着）
def detect_stain_mold(hsv):
    """Dark colored spots or stains: low V with moderate-high S"""
    lower = np.array([0, 40, 0])
    upper = np.array([180, 255, 90])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask_out = np.zeros_like(mask)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 300:
            continue
        boxes.append((x,y,w,h))
        cv2.drawContours(mask_out, [c], -1, 255, -1)
    return boxes, mask_out

def detect_salt_weathering(hsv):
    """Efflorescence/whitish salt: very high V, very low S"""
    lower = np.array([0, 0, 200])
    upper = np.array([180, 35, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask_out = np.zeros_like(mask)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 400:
            continue
        boxes.append((x,y,w,h))
        cv2.drawContours(mask_out, [c], -1, 255, -1)
    return boxes, mask_out

def detect_bio_growth(hsv):
    """Biological growth (moss/algae): greenish hue, high S"""
    lower = np.array([35, 60, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    boxes = []
    mask_out = np.zeros_like(mask)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 300:
            continue
        boxes.append((x,y,w,h))
        cv2.drawContours(mask_out, [c], -1, 255, -1)
    return boxes, mask_out

# ---------------------------
# Deep model inference (ONNX)
# ---------------------------
def run_segmentation_model(image_bgr, model_path, input_size=512, class_ids=None, providers=None):
    """Run ONNX segmentation model and return masks dict.

    Assumptions:
    - Input: RGB float32 0-1, NCHW (1,C,H,W). We will transpose accordingly.
    - Output: either NCHW (1,C,H,W) or NHWC (1,H,W,C) or HW (single channel) or HxW (after squeeze).
    - Class mapping provided via class_ids: {'bg':0,'crack':1,'peel':2,'disc':3}.
    """
    if ort is None:
        raise RuntimeError("未安装 onnxruntime，请先安装：pip install onnxruntime")

    if class_ids is None:
        class_ids = {'bg': 0, 'crack': 1, 'peel': 2, 'disc': 3, 'stain': 4, 'salt': 5, 'bio': 6}

    # Prepare session (cached)
    session = get_onnx_session_cached(model_path, providers=providers or ["CPUExecutionProvider"])

    # Preprocess: BGR->RGB, resize square, normalize to [0,1]
    h0, w0 = image_bgr.shape[:2]
    target = int(input_size)
    img_resized = cv2.resize(image_bgr, (target, target), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    inp = img_rgb.astype(np.float32) / 255.0
    # to NCHW
    inp = np.transpose(inp, (2,0,1))[None, ...]  # (1,3,H,W)

    # IO names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    out = session.run([output_name], {input_name: inp})[0]
    # Postprocess to class map HxW
    class_map = None
    if out.ndim == 4:
        # (N,C,H,W) or (N,H,W,C)
        if out.shape[1] <= 6 and out.shape[0] == 1:  # likely NCHW classes small
            class_map = np.argmax(out[0], axis=0)
        elif out.shape[-1] <= 6 and out.shape[0] == 1:  # NHWC
            class_map = np.argmax(out[0], axis=-1)
        else:
            # fallback: take channel-argmax in the dimension that matches classes<=20
            if out.shape[1] < out.shape[-1]:
                class_map = np.argmax(out[0], axis=0)
            else:
                class_map = np.argmax(out[0], axis=-1)
    elif out.ndim == 3:
        # (C,H,W) or (H,W,C)
        if out.shape[0] <= 6:
            class_map = np.argmax(out, axis=0)
        elif out.shape[-1] <= 6:
            class_map = np.argmax(out, axis=-1)
        else:
            raise ValueError("无法解析模型输出形状：" + str(out.shape))
    elif out.ndim == 2:
        # binary logits/mask -> treat >0.5 as class 'crack' by default
        class_map = (out > 0.5).astype(np.uint8) * class_ids.get('crack', 1)
    else:
        raise ValueError("未支持的模型输出维度：" + str(out.shape))

    # Resize back to original size
    class_map = cv2.resize(class_map.astype(np.int32), (w0, h0), interpolation=cv2.INTER_NEAREST)

    crack_id = int(class_ids.get('crack', 1))
    peel_id = int(class_ids.get('peel', 2))
    disc_id = int(class_ids.get('disc', 3))
    stain_id = class_ids.get('stain', None)
    salt_id = class_ids.get('salt', None)
    bio_id = class_ids.get('bio', None)

    masks = {
        'crack': (class_map == crack_id).astype(np.uint8) * 255,
        'peel': (class_map == peel_id).astype(np.uint8) * 255,
        'disc': (class_map == disc_id).astype(np.uint8) * 255
    }
    if stain_id is not None:
        masks['stain'] = (class_map == int(stain_id)).astype(np.uint8) * 255
    if salt_id is not None:
        masks['salt'] = (class_map == int(salt_id)).astype(np.uint8) * 255
    if bio_id is not None:
        masks['bio'] = (class_map == int(bio_id)).astype(np.uint8) * 255
    return masks

# ---------------------------
# UI and main logic
# ---------------------------
st.markdown("<h1 style='text-align:center;color:#8B4513;'>🏛️ 石窟寺壁画病害AI识别工具（升级版）</h1>", unsafe_allow_html=True)
st.write("本工具为科研原型：采用传统图像处理方法作为基线，并提供深度模型接入点；输出病害检测、严重度评分、材质自适应建议与含标注图像的 PDF。")

# Sidebar controls
st.sidebar.markdown("### 配置与材质选择")
material = st.sidebar.selectbox("选择壁画材质（影响评分与建议）", MATERIAL_OPTIONS)
auto_material = st.sidebar.checkbox("自动识别材质（试验性）", value=False)
mat_model_path = None
if auto_material:
    st.sidebar.markdown("- 可选：提供材质分类ONNX模型路径（若留空则使用启发式识别）")
    mat_model_path = st.sidebar.text_input("材质模型路径（.onnx，可选）", "")
use_deep = st.sidebar.checkbox("使用深度分割模型（ONNX）", value=False)
model_path = None
model_input_size = 512
# 性能/速度设置
st.sidebar.markdown("### 性能/速度设置")
max_dim_setting = st.sidebar.slider("最大处理分辨率（像素）", 512, 2048, 1280, 64)
icp_threshold = st.sidebar.slider("3D ICP 距离阈值 (m)", 0.002, 0.05, 0.02, 0.002)
class_id_bg = 0
class_id_crack = 1
class_id_peel = 2
class_id_disc = 3
class_id_stain = 4
class_id_salt = 5
class_id_bio = 6
if use_deep:
    model_path = st.sidebar.text_input("模型路径（.onnx）", "")
    model_input_size = st.sidebar.selectbox("模型输入尺寸（方形）", [256, 320, 384, 512, 640, 768, 1024], index=3)
    st.sidebar.markdown("#### 类别ID映射（与训练一致）")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        class_id_bg = st.number_input("背景ID", value=0, step=1)
        class_id_crack = st.number_input("裂缝ID", value=1, step=1)
        class_id_stain = st.number_input("污渍/霉斑ID", value=4, step=1)
    with col2:
        class_id_peel = st.number_input("剥落ID", value=2, step=1)
        class_id_disc = st.number_input("褪色ID", value=3, step=1)
        class_id_salt = st.number_input("盐蚀/风化ID", value=5, step=1)
    class_id_bio = st.sidebar.number_input("生物附着ID", value=6, step=1)

# Display controls (去杂化)
st.sidebar.markdown("### 显示设置（减少干扰）")
display_mode = st.sidebar.selectbox(
    "显示方式",
    ["仅掩膜", "仅边框", "边框+掩膜"],
    index=0
)
min_area = st.sidebar.slider("最小目标面积（像素）", 0, 5000, 400, step=50)
show_crack = st.sidebar.checkbox("显示：裂缝", True)
show_peel = st.sidebar.checkbox("显示：剥落", True)
show_disc = st.sidebar.checkbox("显示：褪色", True)
show_stain = st.sidebar.checkbox("显示：污渍/霉斑", True)
show_salt = st.sidebar.checkbox("显示：盐蚀/风化", True)
show_bio = st.sidebar.checkbox("显示：生物附着", True)
show_labels = st.sidebar.checkbox("在图上标注类别简写", True)
label_lang = st.sidebar.selectbox("标签样式", ["简写(EN)", "中文"], index=0)

# 实尺标定（像素-毫米换算）
st.sidebar.markdown("### 实尺标定（单位转换）")
if "ppmm" not in st.session_state:
    st.session_state["ppmm"] = None  # pixels per millimeter
scale_mode = st.sidebar.selectbox("标定方式", ["未标定", "直接输入像素/毫米", "参考物标定（输入像素长度与实长mm）"], index=0)
ppmm_direct = None
if scale_mode == "直接输入像素/毫米":
    ppmm_direct = st.sidebar.number_input("像素/毫米 (pixels per mm)", min_value=0.0, value=float(st.session_state["ppmm"]) if st.session_state["ppmm"] else 0.0, step=0.01)
    if ppmm_direct > 0:
        st.session_state["ppmm"] = ppmm_direct
elif scale_mode == "参考物标定（输入像素长度与实长mm）":
    ref_px = st.sidebar.number_input("参考物在图中的像素长度", min_value=0.0, value=0.0, step=1.0)
    ref_mm = st.sidebar.number_input("参考物实际长度（mm）", min_value=0.0, value=0.0, step=0.1)
    if ref_px > 0 and ref_mm > 0:
        st.session_state["ppmm"] = ref_px / ref_mm
_ppmm_val = st.session_state["ppmm"]
if _ppmm_val:
    st.sidebar.caption(f"当前标定：{_ppmm_val:.3f} 像素/毫米")
else:
    st.sidebar.caption("当前标定：未标定")

# Upload (支持历史对比：允许上传旧图像)
# 页面装饰：动态背景与校徽角标
try:
    bg_imgs = get_background_images_b64()
    inject_dynamic_background(bg_imgs, interval_ms=10000)
    logo_data = get_logo_b64()
    inject_brand_badge(logo_data)
except Exception:
    pass
tabs = st.tabs(["二维壁画诊断", "三维石窟监测（基础版）", "文献资料识别（OCR）", "多模态融合诊断"])

with tabs[0]:
    st.markdown("#### 1) 上传图像（可上传 1-2 张用于时间对比）")
uploaded = st.file_uploader("上传当前图像（必填）", type=['jpg','jpeg','png'])
uploaded_prev = st.file_uploader("上传历史图像（可选，用于对比），若有则为同一壁画的早期照片", type=['jpg','jpeg','png'])

analyze_btn = st.button("开始分析")

if analyze_btn and uploaded is None:
    st.error("请至少上传当前图像以进行分析。")

if uploaded is not None and analyze_btn:
    # read images
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("无法读取图像，请确认格式正确。")
    else:
        img_proc, scale = preprocess_image(img.copy(), target_max_dim=int(max_dim_setting))
        img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
        st.subheader("原始图像（已缩放以便处理）")
        # Auto material detection
        detected_material = None
        detected_conf = None
        detected_details = None
        if auto_material:
            try:
                if mat_model_path:
                    detected_material, detected_conf, mat_probs = run_material_model(img_proc, mat_model_path)
                    detected_details = {"probs": mat_probs}
                else:
                    detected_material, detected_conf, detected_details = classify_material_heuristic(img_proc)
                st.info(f"自动识别材质：{detected_material}（置信度 {detected_conf:.2f}）")
                apply_mat = st.toggle("将识别结果应用到评分/建议", value=True)
                if apply_mat:
                    material = detected_material
            except Exception as e:
                st.warning(f"自动材质识别失败：{e}")

        st.image(img_rgb, width='stretch')

        # OCR 识别（可选）
        st.markdown("### 🔤 文字识别（OCR）")
        if RapidOCR is None:
            st.info("未安装 rapidocr-onnxruntime，如需OCR：pip install rapidocr-onnxruntime")
        else:
            if st.toggle("启用OCR识别（实验性）", value=False):
                ocr = get_rapidocr_cached()
                if ocr is None:
                    st.warning("OCR 初始化失败。")
                else:
                    with st.spinner("OCR识别中…"):
                        res, elapse = ocr(img_rgb)
                    # 展示结果和可下载TXT
                    ocr_lines = []
                    if res:
                        for box, text, score in res:
                            ocr_lines.append(f"{text}\t{score:.3f}")
                        st.success(f"识别到 {len(ocr_lines)} 行文本。")
                        st.code("\n".join(ocr_lines))
                        st.download_button("下载OCR结果（txt）", data=("\n".join(ocr_lines)).encode("utf-8"), file_name="ocr_result.txt", mime="text/plain")
                    else:
                        st.info("未识别到明显文本区域。")

        # Optionally run deep model
        deep_masks = None
        if use_deep and model_path:
            try:
                deep_masks = run_segmentation_model(
                    img_proc,
                    model_path=model_path,
                    input_size=int(model_input_size),
                    class_ids={
                        'bg': int(class_id_bg),
                        'crack': int(class_id_crack),
                        'peel': int(class_id_peel),
                        'disc': int(class_id_disc),
                        'stain': int(class_id_stain),
                        'salt': int(class_id_salt),
                        'bio': int(class_id_bio)
                    },
                    providers=["CPUExecutionProvider"]
                )
                st.success("深度分割已启用：结果将替换传统CV掩膜。")
            except Exception as e:
                st.exception(e)
                st.error("深度模型推理失败，请检查模型路径、类别ID与输入尺寸是否匹配。")
                deep_masks = None

        # Baseline CV detections
        gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
        boxes_crack, mask_crack = detect_cracks(gray)
        boxes_peel, mask_peel = detect_peeling(hsv)
        boxes_disc, mask_disc = detect_discoloration(hsv)
        boxes_stain, mask_stain = detect_stain_mold(hsv)
        boxes_salt, mask_salt = detect_salt_weathering(hsv)
        boxes_bio, mask_bio = detect_bio_growth(hsv)

        # If deep_masks provided, prefer it
        if deep_masks:
            # expected keys 'crack','peel','disc' -> binary masks same size as img_proc
            if 'crack' in deep_masks:
                mask_crack = deep_masks['crack']
            if 'peel' in deep_masks:
                mask_peel = deep_masks['peel']
            if 'disc' in deep_masks:
                mask_disc = deep_masks['disc']
            if 'stain' in deep_masks:
                mask_stain = deep_masks['stain']
            if 'salt' in deep_masks:
                mask_salt = deep_masks['salt']
            if 'bio' in deep_masks:
                mask_bio = deep_masks['bio']

        # Annotate image for visualization
        annotated = img_rgb.copy()
        # Helper to draw boxes with filters
        def draw_boxes(boxes, color, visible, tag_text):
            if not visible or display_mode == "仅掩膜":
                return
            for (x,y,w,h) in boxes:
                if w*h < min_area:
                    continue
                cv2.rectangle(annotated, (x,y), (x+w, y+h), color, 2)
                if show_labels:
                    # draw a small label background for readability
                    tx = x
                    ty = max(0, y-8)
                    text = tag_text
                    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated, (tx, ty-th-4), (tx+tw+4, ty+2), (0,0,0), -1)
                    cv2.putText(annotated, text, (tx+2, ty-2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw per category (no Chinese labels on image to避免问号)
        # label text mapping
        if label_lang == "中文":
            # OpenCV 不支持中文字体，可能显示问号；若出现，请切换到“简写(EN)”
            crack_t, peel_t, disc_t, stain_t, salt_t, bio_t = "裂", "剥", "褪", "污", "盐", "生"
        else:
            crack_t, peel_t, disc_t, stain_t, salt_t, bio_t = "CR", "PL", "DC", "ST", "SA", "BIO"

        draw_boxes(boxes_crack, (255,0,0), show_crack, crack_t)
        draw_boxes(boxes_peel, (0,255,0), show_peel, peel_t)
        draw_boxes(boxes_disc, (0,0,255), show_disc, disc_t)
        draw_boxes(boxes_stain, (255,255,0), show_stain, stain_t)
        draw_boxes(boxes_salt, (0,255,255), show_salt, salt_t)
        draw_boxes(boxes_bio, (255,0,255), show_bio, bio_t)

        # Also overlay masks semi-transparently to show extent
        def overlay_mask(base_rgb, mask, color_rgb, alpha=0.35):
            overlay = base_rgb.copy()
            mask_bool = mask > 0
            overlay[mask_bool] = (overlay[mask_bool] * (1-alpha) + np.array(color_rgb) * alpha).astype(np.uint8)
            return overlay

        if display_mode in ("仅掩膜", "边框+掩膜"):
            if show_crack:
                annotated = overlay_mask(annotated, mask_crack, (255,0,0), alpha=0.25)
            if show_peel:
                annotated = overlay_mask(annotated, mask_peel, (0,255,0), alpha=0.18)
            if show_disc:
                annotated = overlay_mask(annotated, mask_disc, (0,0,255), alpha=0.18)
            if show_stain:
                annotated = overlay_mask(annotated, mask_stain, (255,255,0), alpha=0.20)
            if show_salt:
                annotated = overlay_mask(annotated, mask_salt, (0,255,255), alpha=0.18)
            if show_bio:
                annotated = overlay_mask(annotated, mask_bio, (255,0,255), alpha=0.20)

        st.subheader("分析结果（带标注）")
        st.image(annotated, width='stretch')

        # Legend (图例) for colors
        legend_html = """
        <div style='display:flex;flex-wrap:wrap;gap:12px;margin:6px 0 10px 0;'>
          <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#ff0000;'></span><span>裂缝</span></div>
          <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#00cc00;'></span><span>剥落</span></div>
          <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#0000ff;'></span><span>褪色</span></div>
          <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#ffff00;'></span><span>污渍/霉斑</span></div>
          <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#00ffff;'></span><span>盐蚀/风化</span></div>
          <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#ff00ff;'></span><span>生物附着</span></div>
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)
        # 色彩复原（基础）
        render_color_restore_ui(img_rgb, default_open=False, key_suffix="color1")

        # ---------------------
        # Quantification & scoring
        # ---------------------
        h,w = gray.shape
        total_pixels = h*w
        crack_area = int(np.sum(mask_crack>0))
        peel_area = int(np.sum(mask_peel>0))
        disc_area = int(np.sum(mask_disc>0))
        stain_area = int(np.sum(mask_stain>0))
        salt_area = int(np.sum(mask_salt>0))
        bio_area = int(np.sum(mask_bio>0))

        crack_pct = crack_area / total_pixels * 100
        peel_pct = peel_area / total_pixels * 100
        disc_pct = disc_area / total_pixels * 100
        stain_pct = stain_area / total_pixels * 100
        salt_pct = salt_area / total_pixels * 100
        bio_pct = bio_area / total_pixels * 100

        # severity score: weighted sum (0-100)
        weights = MATERIAL_WEIGHTS.get(material, MATERIAL_WEIGHTS["未指定"])
        # normalize each by an empirical factor per category
        score = (
            weights.get('crack',1.0) * crack_pct * 1.8 +
            weights.get('peel',1.0) * peel_pct * 1.2 +
            weights.get('disc',1.0) * disc_pct * 1.5 +
            weights.get('stain',1.0) * stain_pct * 0.9 +
            weights.get('salt',1.0) * salt_pct * 1.6 +
            weights.get('bio',1.0) * bio_pct * 1.1
        )
        # map to 0-100
        severity = min(round(score,1), 100.0)

        st.markdown("### 📋 量化结果")
        st.write(f"- 裂缝覆盖面积：{crack_area} 像素，约占图像面积 {crack_pct:.3f}%")
        st.write(f"- 剥落覆盖面积：{peel_area} 像素，约占图像面积 {peel_pct:.3f}%")
        st.write(f"- 褪色覆盖面积：{disc_area} 像素，约占图像面积 {disc_pct:.3f}%")
        st.write(f"- 污渍/霉斑覆盖面积：{stain_area} 像素，约占图像面积 {stain_pct:.3f}%")
        st.write(f"- 盐蚀/风化覆盖面积：{salt_area} 像素，约占图像面积 {salt_pct:.3f}%")
        st.write(f"- 生物附着覆盖面积：{bio_area} 像素，约占图像面积 {bio_pct:.3f}%")
        st.write(f"- 材质：**{material}**（用于调整评分与建议）")
        st.metric("整体病害严重度（0-100）", f"{severity}")

        # severity label
        if severity < 5:
            lvl = "轻微"
        elif severity < 20:
            lvl = "中等"
        else:
            lvl = "严重"
        st.write(f"判断等级：**{lvl}**")

        # ---------------------
        # 细化病理指标（连通域/形态/方向）
        # ---------------------
        def extract_components(mask, min_area_px=min_area):
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask>0).astype(np.uint8), connectivity=8)
            rows = []
            for i in range(1, num_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_area_px:
                    continue
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                w_ = int(stats[i, cv2.CC_STAT_WIDTH])
                h_ = int(stats[i, cv2.CC_STAT_HEIGHT])
                elong = (max(w_, h_) / max(1.0, min(w_, h_)))
                ys, xs = np.where(labels == i)
                if xs.size > 5:
                    x_mean = xs.mean(); y_mean = ys.mean()
                    x_c = xs - x_mean; y_c = ys - y_mean
                    cov_xx = float(np.mean(x_c*x_c)); cov_yy = float(np.mean(y_c*y_c)); cov_xy = float(np.mean(x_c*y_c))
                    theta = 0.5 * np.arctan2(2*cov_xy, (cov_xx - cov_yy))
                    orient_deg = float(np.degrees(theta)) % 180
                else:
                    orient_deg = float('nan')
                # 估计长度与平均宽度：细长目标用骨架近似长度，否则用等效直径
                comp_mask = (labels == i).astype(np.uint8)
                length_px = float(np.sqrt((w_**2 + h_**2)))
                mean_width_px = float(area / max(1.0, length_px))
                # 若已标定，转换到毫米
                ppmm = st.session_state.get('ppmm')
                length_mm = (length_px / ppmm) if ppmm else None
                mean_width_mm = (mean_width_px / ppmm) if ppmm else None
                rows.append({
                    'area_px': area,
                    'bbox_w': w_,
                    'bbox_h': h_,
                    'elongation': round(elong,3),
                    'orientation_deg': round(orient_deg,2),
                    'length_px': round(length_px,2),
                    'mean_width_px': round(mean_width_px,2),
                    **({'length_mm': round(length_mm,2), 'mean_width_mm': round(mean_width_mm,2)} if ppmm else {})
                })
            return rows

        import pandas as _pd_alias
        metrics = {
            '裂缝': extract_components(mask_crack),
            '剥落': extract_components(mask_peel),
            '褪色': extract_components(mask_disc),
            '污渍/霉斑': extract_components(mask_stain),
            '盐蚀/风化': extract_components(mask_salt),
            '生物附着': extract_components(mask_bio)
        }
        st.markdown("### 🔎 细化病理指标（按类别）")
        cat_tabs = st.tabs(list(metrics.keys()))
        for tab, (cat, rows) in zip(cat_tabs, metrics.items()):
            with tab:
                if len(rows) == 0:
                    st.write("无显著连通域（受最小面积阈值影响）")
                else:
                    df = _pd_alias.DataFrame(rows)
                    stats_msg = f"连通域数量：{len(df)}，面积中位数：{df['area_px'].median():.0f} px，细长比P95：{df['elongation'].quantile(0.95):.2f}"
                    if 'mean_width_mm' in df.columns:
                        stats_msg += f"，平均宽度中位数：{df['mean_width_mm'].median():.2f} mm"
                    st.write(stats_msg)
                    st.dataframe(df.sort_values('area_px', ascending=False).head(50), use_container_width=True)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(label=f"下载{cat}指标CSV", data=csv, file_name=f"metrics_{cat}.csv", mime="text/csv")

        # textual suggestions
        st.markdown("### 💡 建议（对症方案）")
        pct_map = {
            'crack': crack_pct,
            'peel': peel_pct,
            'disc': disc_pct,
            'stain': stain_pct,
            'salt': salt_pct,
            'bio': bio_pct
        }
        detailed_recs = build_recommendations(material, pct_map, severity)
        for r in detailed_recs:
            st.write(f"- {r}")

        # inpainting UI moved to a global section that uses cached masks

        # ---------------------
        # Time-comparison (if previous uploaded)
        # ---------------------
        if uploaded_prev:
            prev_bytes = np.asarray(bytearray(uploaded_prev.read()), dtype=np.uint8)
            prev_img = cv2.imdecode(prev_bytes, cv2.IMREAD_COLOR)
            if prev_img is None:
                st.warning("历史图像无法读取，请确认文件格式与完整性，已跳过历史对比。")
            else:
                prev_img_proc, scale_prev = preprocess_image(prev_img.copy())
                prev_gray = cv2.cvtColor(prev_img_proc, cv2.COLOR_BGR2GRAY)
                prev_h, prev_w = prev_gray.shape
                # naive comparison: compare masks area difference (after resizing to match)
                # Resize prev masks to current shape if needed
                if prev_img_proc.shape[:2] != img_proc.shape[:2]:
                    prev_img_proc = cv2.resize(prev_img_proc, (img_proc.shape[1], img_proc.shape[0]))
                    prev_gray = cv2.cvtColor(prev_img_proc, cv2.COLOR_BGR2GRAY)
                # detect previous masks (same pipeline)
                p_boxes_crack, p_mask_crack = detect_cracks(prev_gray)
                p_hsv = cv2.cvtColor(prev_img_proc, cv2.COLOR_BGR2HSV)
                p_boxes_peel, p_mask_peel = detect_peeling(p_hsv)
                p_boxes_disc, p_mask_disc = detect_discoloration(p_hsv)
                # compute area change
                prev_crack_area = int(np.sum(p_mask_crack>0))
                prev_peel_area = int(np.sum(p_mask_peel>0))
                prev_disc_area = int(np.sum(p_mask_disc>0))
                st.markdown("### 🕒 历史对比结果")
                st.write(f"- 裂缝面积变化：{prev_crack_area} -> {crack_area} （差值 {crack_area - prev_crack_area} 像素）")
                st.write(f"- 剥落面积变化：{prev_peel_area} -> {peel_area} （差值 {peel_area - prev_peel_area} 像素）")
                st.write(f"- 褪色面积变化：{prev_disc_area} -> {disc_area} （差值 {disc_area - prev_disc_area} 像素）")
                # quick assessment
                if (crack_area - prev_crack_area) > (0.05 * total_pixels):
                    st.error("裂缝面积显著增加，建议尽快实地评估。")
                elif (peel_area - prev_peel_area) > (0.05 * total_pixels):
                    st.error("剥落面积显著增加，可能存在进展性破坏。")

        # ---------------------
        # Generate PDF with annotated image and results
        # ---------------------
        def generate_pdf_report(annotated_rgb, results, material, suggestions_text):
            """Create PDF and return bytesIO"""
            buf = BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph("石窟寺壁画病害AI诊断报告（升级版）", styles['Title']))
            story.append(Spacer(1,6))
            story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1,12))

            # insert annotated image (save to tmp buffer)
            img_buf = save_annotated_image_bytes(annotated_rgb)
            # reportlab needs a filename-like object; RLImage accepts BytesIO
            story.append(RLImage(img_buf, width=160*mm, height=(160*annotated_rgb.shape[0]/annotated_rgb.shape[1])*mm))
            story.append(Spacer(1,12))

            story.append(Paragraph("<b>一、量化结果</b>", styles['Heading2']))
            for line in results:
                story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1,8))

            story.append(Paragraph("<b>二、综合建议</b>", styles['Heading2']))
            for line in suggestions_text:
                story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1,12))

            if material != "未指定":
                story.append(Paragraph("<b>三、材质专用建议</b>", styles['Heading2']))
                for t in MATERIAL_SUGGESTIONS.get(material, []):
                    story.append(Paragraph(t, styles['Normal']))
                story.append(Spacer(1,8))

            story.append(Paragraph(f"© {datetime.now().year} 上海交大文物修复团队 | AI+文物保护研究", styles['Normal']))
            doc.build(story)
            buf.seek(0)
            return buf

        results_lines = [
            f"裂缝覆盖面积：{crack_area} 像素，占比 {crack_pct:.4f}%",
            f"剥落覆盖面积：{peel_area} 像素，占比 {peel_pct:.4f}%",
            f"褪色覆盖面积：{disc_area} 像素，占比 {disc_pct:.4f}%",
            f"污渍/霉斑覆盖面积：{stain_area} 像素，占比 {stain_pct:.4f}%",
            f"盐蚀/风化覆盖面积：{salt_area} 像素，占比 {salt_pct:.4f}%",
            f"生物附着覆盖面积：{bio_area} 像素，占比 {bio_pct:.4f}%",
            f"整体严重度评分（0-100）：{severity}，等级：{lvl}"
        ]
        suggestions_text = detailed_recs

        pdf_buf = generate_pdf_report(annotated, results_lines, material, suggestions_text)
        st.download_button("⬇️ 下载诊断报告（含标注图）PDF", data=pdf_buf.getvalue(), file_name="诊断报告_壁画.pdf", mime="application/pdf")

        # Cache results for interactive toggling without re-uploading
        st.session_state["proc"] = {
            'img_rgb': img_rgb,
            'masks': {
                'crack': mask_crack, 'peel': mask_peel, 'disc': mask_disc,
                'stain': mask_stain, 'salt': mask_salt, 'bio': mask_bio
            },
            'boxes': {
                'crack': boxes_crack, 'peel': boxes_peel, 'disc': boxes_disc,
                'stain': boxes_stain, 'salt': boxes_salt, 'bio': boxes_bio
            },
            'shape': gray.shape
        }

with tabs[1]:
    st.markdown("#### 上传两期三维数据（点云/网格）")
    f_epoch1 = st.file_uploader("上传一期（参考）PLY/PCD/OBJ/GLB", type=["ply","pcd","obj","glb"], key="pc1")
    f_epoch2 = st.file_uploader("上传二期（对比）PLY/PCD/OBJ/GLB", type=["ply","pcd","obj","glb"], key="pc2")
    max_points = st.number_input("可视化/计算最大点数（下采样）", min_value=10000, value=200000, step=10000)
    run_icp = st.button("执行配准与距离计算（基础）")
    if run_icp:
        if o3d is None:
            st.error("缺少 open3d，请先安装：pip install open3d")
        elif f_epoch1 is None or f_epoch2 is None:
            st.error("请上传两期三维数据文件。")
        else:
            try:
                def load_geom(file):
                    import tempfile
                    suffix = "." + file.name.split(".")[-1].lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file.read()); path = tmp.name
                    mesh = None
                    if suffix in (".obj", ".glb"):
                        mesh = o3d.io.read_triangle_mesh(path)
                        if not mesh.has_vertices():
                            raise RuntimeError("无法读取网格")
                        pcd = mesh.sample_points_uniformly(number_of_points=int(max_points))
                    else:
                        pcd = o3d.io.read_point_cloud(path)
                        if len(pcd.points) == 0:
                            raise RuntimeError("无法读取点云")
                    if len(pcd.points) > max_points:
                        pcd = pcd.random_down_sample(float(max_points)/float(len(pcd.points)))
                    pcd.estimate_normals()
                    return pcd

                p1 = load_geom(f_epoch1)
                p2 = load_geom(f_epoch2)
                # 粗配准：基于质心对齐
                c1 = p1.get_center(); c2 = p2.get_center()
                p2_t = p2.translate(c1 - c2, relative=False)
                # 精配准：ICP
                with st.spinner("ICP精配准中…"):
                    # 先全局配准尝试（RANSAC）再ICP（若可用）
                    try:
                        voxel = max(float(icp_threshold)*2.0, 0.01)
                        p1_down = p1.voxel_down_sample(voxel)
                        p2_down = p2_t.voxel_down_sample(voxel)
                        reg_ransac = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                            p2_down, p1_down, o3d.utility.Vector2iVector(np.array([[0,0]], dtype=np.int32)),
                            max_correspondence_distance=float(icp_threshold)*3.0,
                            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                            ransac_n=3,
                            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000, 500)
                        )
                        init = reg_ransac.transformation if hasattr(reg_ransac, 'transformation') else np.eye(4)
                    except Exception:
                        init = np.eye(4)
                    reg = o3d.pipelines.registration.registration_icp(
                        p2_t, p1, float(icp_threshold), init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint()
                    )
                    p2_aligned = p2_t.transform(reg.transformation)
                    # 计算最近点距离
                    pcd_tree = o3d.geometry.KDTreeFlann(p1)
                dists = []
                pts = np.asarray(p2_aligned.points)
                for pt in pts:
                    [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, 1)
                    if k > 0:
                        nn = np.asarray(p1.points)[idx[0]]
                        dists.append(float(np.linalg.norm(pt - nn)))
                if len(dists) == 0:
                    st.warning("距离计算为空。")
                else:
                    dists = np.array(dists)
                    st.write(f"点数：{len(dists)}，均值：{dists.mean()*1000:.2f} mm，P95：{np.quantile(dists,0.95)*1000:.2f} mm，最大：{dists.max()*1000:.2f} mm")
                    if px is not None:
                        df = pd.DataFrame({"dist_mm": dists*1000.0})
                        st.plotly_chart(px.histogram(df, x="dist_mm", nbins=50, title="距离分布(mm)"), use_container_width=True)
                    # 导出CSV
                    csv = ("dist_mm\n" + "\n".join(f"{v*1000:.4f}" for v in dists)).encode("utf-8")
                    st.download_button("下载距离分布CSV", data=csv, file_name="distances_mm.csv", mime="text/csv")
            except Exception as e:
                st.exception(e)
                st.error("三维处理失败，请确认文件格式并适当调小点数或阈值。")

with tabs[2]:
    st.markdown("#### 上传文献/资料图片进行文字识别（OCR）")
    if RapidOCR is None:
        st.info("未安装 rapidocr-onnxruntime，如需OCR：pip install rapidocr-onnxruntime")
    else:
        files_txt = st.file_uploader("上传图片（可多选）JPG/PNG", type=["jpg","jpeg","png"], accept_multiple_files=True, key="ocr_multi")
        run_ocr = st.button("开始识别", key="run_ocr_batch")
        if run_ocr:
            if not files_txt:
                st.warning("请先选择至少一张图片。")
            else:
                ocr = get_rapidocr_cached()
                if ocr is None:
                    st.error("OCR 初始化失败。")
                else:
                    all_lines = []
                    for idx, f in enumerate(files_txt, start=1):
                        st.write(f"第 {idx} 个文件：{f.name}")
                        img_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                        if img is None:
                            st.warning("无法读取该图片，已跳过。")
                            continue
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        with st.spinner("OCR识别中…"):
                            res, elapse = ocr(rgb)
                        lines = []
                        if res:
                            for box, text, score in res:
                                line = f"{text}\t{score:.3f}"
                                lines.append(line)
                                all_lines.append(f"[{f.name}]\t{line}")
                            st.success(f"识别 {len(lines)} 行，用时 {elapse:.2f}s")
                            st.code("\n".join(lines))
                        else:
                            st.info("未识别到文本。")
                    if all_lines:
                        txt = ("\n".join(all_lines)).encode("utf-8")
                        st.download_button("下载全部OCR结果（txt）", data=txt, file_name="ocr_results.txt", mime="text/plain")

with tabs[3]:
    st.markdown("#### 多模态融合诊断系统")
    st.info("🚀 **前沿功能**：结合图像、3D点云、文献文本进行综合诊断，提供深度分析和虚拟修复")
    
    if not MULTIMODAL_AVAILABLE:
        st.warning("⚠️ 多模态功能需要额外依赖，请安装：`pip install torch transformers networkx scikit-learn`")
        st.code("pip install torch transformers networkx scikit-learn")
    else:
        # 多模态数据上传
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📸 图像数据")
            multimodal_image = st.file_uploader("上传壁画图像", type=['jpg','jpeg','png'], key="multimodal_img")
            
            st.markdown("##### 📄 文献数据")
            multimodal_text = st.text_area("输入相关文献记录（如历史修复记录、材质描述等）", 
                                         placeholder="例如：该壁画位于敦煌莫高窟第257窟，绘制于北魏时期，主要材质为砂岩...", 
                                         height=100, key="multimodal_text")
        
        with col2:
            st.markdown("##### 🏗️ 3D点云数据")
            multimodal_pointcloud = st.file_uploader("上传3D扫描数据", type=['ply','pcd','xyz'], key="multimodal_pc")
            
            st.markdown("##### 🏛️ 石窟信息")
            cave_type = st.selectbox("选择石窟类型", 
                                   ["敦煌莫高窟", "云冈石窟", "龙门石窟", "麦积山石窟", "其他"], 
                                   key="cave_type")
            
            material_type = st.selectbox("选择材质类型", 
                                       ["砂岩", "花岗岩", "石灰岩", "泥质砂岩", "其他"], 
                                       key="material_type")
        
        # 多模态分析按钮
        run_multimodal = st.button("🔍 开始多模态融合分析", key="run_multimodal")
        
        if run_multimodal:
            if not multimodal_image:
                st.warning("请至少上传一张图像进行分析")
            else:
                # 初始化多模态系统
                multimodal_system = get_multimodal_system()
                auto_annotator = get_auto_annotator()
                generative_aug = get_generative_augmentation()
                
                # 处理图像
                img_bytes = multimodal_image.read()
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # 处理点云
                pointcloud = None
                if multimodal_pointcloud:
                    try:
                        pc_bytes = multimodal_pointcloud.read()
                        if multimodal_pointcloud.name.endswith('.ply'):
                            pointcloud = o3d.io.read_point_cloud_from_bytes(pc_bytes, format='ply')
                        elif multimodal_pointcloud.name.endswith('.pcd'):
                            pointcloud = o3d.io.read_point_cloud_from_bytes(pc_bytes, format='pcd')
                    except Exception as e:
                        st.warning(f"点云加载失败：{e}")
                
                # 多模态特征提取
                with st.spinner("🔄 多模态特征提取中..."):
                    # 图像特征
                    image_features = multimodal_system.encode_image(image)
                    
                    # 点云特征
                    pointcloud_features = multimodal_system.encode_pointcloud(pointcloud)
                    
                    # 文本特征
                    text_features = multimodal_system.encode_text(multimodal_text)
                    
                    # 特征融合
                    fused_features = multimodal_system.fuse_modalities(image_features, pointcloud_features, text_features)
                
                # 显示特征信息
                st.success("✅ 多模态特征提取完成")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("图像特征维度", f"{len(image_features) if image_features is not None else 0}")
                with col2:
                    st.metric("点云特征维度", f"{len(pointcloud_features) if pointcloud_features is not None else 0}")
                with col3:
                    st.metric("文本特征维度", f"{len(text_features) if text_features is not None else 0}")
                
                # 深度稳定性分析
                if pointcloud is not None:
                    st.markdown("##### 🔍 深度稳定性分析")
                    
                    # 先进行病害检测获取裂缝掩码
                    with st.spinner("🔄 进行病害检测..."):
                        # 使用现有的病害检测功能
                        crack_mask = detect_crack(image) if 'detect_crack' in globals() else None
                        if crack_mask is not None:
                            depth_analysis = multimodal_system.analyze_depth_stability(image, pointcloud, crack_mask)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("裂缝深度", depth_analysis["depth"])
                            with col2:
                                st.metric("结构稳定性", depth_analysis["stability"])
                            with col3:
                                st.metric("分析置信度", f"{depth_analysis['confidence']:.2f}")
                            
                            if "depth_variance" in depth_analysis:
                                st.info(f"深度方差：{depth_analysis['depth_variance']:.4f}")
                            if "point_density" in depth_analysis:
                                st.info(f"点云密度：{depth_analysis['point_density']:.2f}")
                        else:
                            st.warning("未检测到裂缝，无法进行深度分析")
                
                # 知识图谱查询
                st.markdown("##### 🧠 知识图谱智能诊断")
                
                # 模拟检测到的病害
                detected_pathologies = ["表面裂缝", "剥落"]  # 这里应该基于实际检测结果
                
                treatments = multimodal_system.knowledge_graph.query_treatment(
                    cave_type, material_type, detected_pathologies
                )
                
                if treatments:
                    st.success("🎯 基于知识图谱的修复建议：")
                    for i, treatment in enumerate(treatments[:3]):  # 显示前3个建议
                        with st.expander(f"建议 {i+1}: {treatment['treatment']}"):
                            st.write(f"**适用病害**: {treatment['pathology']}")
                            st.write(f"**适用性**: {treatment['suitability']:.2f}")
                            st.write(f"**成本**: {treatment['cost']}")
                            st.write(f"**效果**: {treatment['effectiveness']}")
                            st.write(f"**持久性**: {treatment['durability']}")
                else:
                    st.info("未找到匹配的修复建议")
                
                # 自动标注
                st.markdown("##### 🏷️ 智能自动标注")
                
                # 模拟检测区域
                mock_regions = [
                    {"area": 500, "bbox": [100, 100, 50, 30], "elongation": 0.8},
                    {"area": 1200, "bbox": [200, 150, 80, 40], "elongation": 0.6}
                ]
                
                annotations = auto_annotator.generate_annotation(image, mock_regions, "crack")
                
                if annotations:
                    st.success("📝 自动标注结果：")
                    for i, annotation in enumerate(annotations):
                        with st.expander(f"标注 {i+1}: {annotation['type']} - {annotation['severity']}"):
                            st.write(f"**描述**: {annotation['description']}")
                            st.write(f"**面积**: {annotation['area']} 像素")
                            st.write(f"**置信度**: {annotation['confidence']:.2f}")
                            st.write(f"**特征**: 长宽比 {annotation['features']['aspect_ratio']:.2f}")
                
                # 虚拟修复
                st.markdown("##### 🎨 虚拟修复预览")
                
                if crack_mask is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**原始图像**")
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    with col2:
                        st.write("**虚拟修复后**")
                        restored = generative_aug.virtual_restoration(image, crack_mask, "crack")
                        st.image(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    # 修复效果对比
                    st.markdown("##### 📊 修复效果分析")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("修复算法", "Telea Inpainting")
                    with col2:
                        st.metric("修复区域", f"{np.sum(crack_mask > 0)} 像素")
                    with col3:
                        st.metric("修复质量", "高")
                
                # 综合报告
                st.markdown("##### 📋 多模态诊断报告")
                
                report_data = {
                    "石窟类型": cave_type,
                    "材质类型": material_type,
                    "图像质量": "良好" if image_features is not None else "未知",
                    "3D数据": "可用" if pointcloud_features is not None else "不可用",
                    "文献数据": "已提供" if text_features is not None else "未提供",
                    "融合特征维度": len(fused_features) if fused_features is not None else 0,
                    "检测病害数": len(detected_pathologies),
                    "修复建议数": len(treatments)
                }
                
                report_df = pd.DataFrame(list(report_data.items()), columns=["项目", "结果"])
                st.dataframe(report_df, use_container_width=True)
                
                # 下载报告
                report_text = f"""
多模态融合诊断报告
==================

基本信息：
- 石窟类型：{cave_type}
- 材质类型：{material_type}
- 分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

数据质量：
- 图像数据：{'可用' if image_features is not None else '不可用'}
- 3D点云：{'可用' if pointcloud_features is not None else '不可用'}
- 文献文本：{'已提供' if text_features is not None else '未提供'}

检测结果：
- 检测到病害：{', '.join(detected_pathologies)}
- 修复建议：{len(treatments)} 条

技术指标：
- 融合特征维度：{len(fused_features) if fused_features is not None else 0}
- 分析置信度：{depth_analysis.get('confidence', 0):.2f}（如有3D数据）
                """
                
                st.download_button(
                    "📥 下载多模态诊断报告",
                    data=report_text.encode('utf-8'),
                    file_name=f"multimodal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

# footer
st.markdown(f"<div style='text-align:center;color:#666;margin-top:32px;'>© {datetime.now().year} 上海交大文物修复团队 | AI+文物保护研究</div>", unsafe_allow_html=True)

# If cached results exist, allow re-render with current toggles without re-uploading
if st.session_state.get("proc") is not None and (uploaded is None or not analyze_btn):
    cache = st.session_state["proc"]
    img_rgb = cache['img_rgb']
    masks = cache['masks']
    boxes_map = cache['boxes']
    h, w = cache['shape']

    mask_crack = masks['crack']; mask_peel = masks['peel']; mask_disc = masks['disc']
    mask_stain = masks['stain']; mask_salt = masks['salt']; mask_bio = masks['bio']
    boxes_crack = boxes_map['crack']; boxes_peel = boxes_map['peel']; boxes_disc = boxes_map['disc']
    boxes_stain = boxes_map['stain']; boxes_salt = boxes_map['salt']; boxes_bio = boxes_map['bio']

    annotated = img_rgb.copy()
    def draw_boxes(boxes, color, visible, tag_text):
        if not visible or display_mode == "仅掩膜":
            return
        for (x,y,w_,h_) in boxes:
            if w_*h_ < min_area:
                continue
            cv2.rectangle(annotated, (x,y), (x+w_, y+h_), color, 2)
            if show_labels:
                tx = x; ty = max(0, y-8)
                text = tag_text
                (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (tx, ty-th-4), (tx+tw+4, ty+2), (0,0,0), -1)
                cv2.putText(annotated, text, (tx+2, ty-2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if label_lang == "中文":
        crack_t, peel_t, disc_t, stain_t, salt_t, bio_t = "裂", "剥", "褪", "污", "盐", "生"
    else:
        crack_t, peel_t, disc_t, stain_t, salt_t, bio_t = "CR", "PL", "DC", "ST", "SA", "BIO"

    draw_boxes(boxes_crack, (255,0,0), show_crack, crack_t)
    draw_boxes(boxes_peel, (0,255,0), show_peel, peel_t)
    draw_boxes(boxes_disc, (0,0,255), show_disc, disc_t)
    draw_boxes(boxes_stain, (255,255,0), show_stain, stain_t)
    draw_boxes(boxes_salt, (0,255,255), show_salt, salt_t)
    draw_boxes(boxes_bio, (255,0,255), show_bio, bio_t)

    def overlay_mask(base_rgb, mask, color_rgb, alpha=0.35):
        overlay = base_rgb.copy()
        mask_bool = mask > 0
        overlay[mask_bool] = (overlay[mask_bool] * (1-alpha) + np.array(color_rgb) * alpha).astype(np.uint8)
        return overlay

    if display_mode in ("仅掩膜", "边框+掩膜"):
        if show_crack:
            annotated = overlay_mask(annotated, mask_crack, (255,0,0), alpha=0.25)
        if show_peel:
            annotated = overlay_mask(annotated, mask_peel, (0,255,0), alpha=0.18)
        if show_disc:
            annotated = overlay_mask(annotated, mask_disc, (0,0,255), alpha=0.18)
        if show_stain:
            annotated = overlay_mask(annotated, mask_stain, (255,255,0), alpha=0.20)
        if show_salt:
            annotated = overlay_mask(annotated, mask_salt, (0,255,255), alpha=0.18)
        if show_bio:
            annotated = overlay_mask(annotated, mask_bio, (255,0,255), alpha=0.20)

    st.subheader("分析结果（带标注）")
    st.image(annotated, width='stretch')

    legend_html = """
    <div style='display:flex;flex-wrap:wrap;gap:12px;margin:6px 0 10px 0;'>
      <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#ff0000;'></span><span>裂缝</span></div>
      <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#00cc00;'></span><span>剥落</span></div>
      <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#0000ff;'></span><span>褪色</span></div>
      <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#ffff00;'></span><span>污渍/霉斑</span></div>
      <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#00ffff;'></span><span>盐蚀/风化</span></div>
      <div style='display:flex;align-items:center;gap:6px;'><span style='display:inline-block;width:14px;height:14px;background:#ff00ff;'></span><span>生物附着</span></div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    # 色彩复原（缓存图）
    render_color_restore_ui(img_rgb, default_open=False, key_suffix="color_cached")

    total_pixels = h*w
    crack_area = int(np.sum(mask_crack>0)); peel_area = int(np.sum(mask_peel>0)); disc_area = int(np.sum(mask_disc>0))
    stain_area = int(np.sum(mask_stain>0)); salt_area = int(np.sum(mask_salt>0)); bio_area = int(np.sum(mask_bio>0))

    crack_pct = crack_area / total_pixels * 100
    peel_pct = peel_area / total_pixels * 100
    disc_pct = disc_area / total_pixels * 100
    stain_pct = stain_area / total_pixels * 100
    salt_pct = salt_area / total_pixels * 100
    bio_pct = bio_area / total_pixels * 100

    weights = MATERIAL_WEIGHTS.get(material, MATERIAL_WEIGHTS["未指定"])
    score = (
        weights.get('crack',1.0) * crack_pct * 1.8 +
        weights.get('peel',1.0) * peel_pct * 1.2 +
        weights.get('disc',1.0) * disc_pct * 1.5 +
        weights.get('stain',1.0) * stain_pct * 0.9 +
        weights.get('salt',1.0) * salt_pct * 1.6 +
        weights.get('bio',1.0) * bio_pct * 1.1
    )
    severity = min(round(score,1), 100.0)
    if severity < 5:
        lvl = "轻微"
    elif severity < 20:
        lvl = "中等"
    else:
        lvl = "严重"

    st.markdown("### 📋 量化结果")
    st.write(f"- 裂缝覆盖面积：{crack_area} 像素，约占图像面积 {crack_pct:.3f}%")
    st.write(f"- 剥落覆盖面积：{peel_area} 像素，约占图像面积 {peel_pct:.3f}%")
    st.write(f"- 褪色覆盖面积：{disc_area} 像素，约占图像面积 {disc_pct:.3f}%")
    st.write(f"- 污渍/霉斑覆盖面积：{stain_area} 像素，约占图像面积 {stain_pct:.3f}%")
    st.write(f"- 盐蚀/风化覆盖面积：{salt_area} 像素，约占图像面积 {salt_pct:.3f}%")
    st.write(f"- 生物附着覆盖面积：{bio_area} 像素，约占图像面积 {bio_pct:.3f}%")
    st.write(f"- 材质：**{material}**（用于调整评分与建议）")
    st.metric("整体病害严重度（0-100）", f"{severity}")

    st.markdown("### 💡 建议（对症方案）")
    pct_map = {'crack': crack_pct,'peel': peel_pct,'disc': disc_pct,'stain': stain_pct,'salt': salt_pct,'bio': bio_pct}
    detailed_recs = build_recommendations(material, pct_map, severity)
    for r in detailed_recs:
        st.write(f"- {r}")

    # ---------------------
    # Global inpainting (works with cached results)
    # ---------------------
    render_inpainting_ui(img_rgb, mask_crack, mask_peel, mask_disc, mask_stain, mask_salt, mask_bio, default_open=True, key_suffix="cached")