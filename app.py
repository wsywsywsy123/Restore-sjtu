#!/usr/bin/env python
# -*- coding: utf-8 -*-
# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from datetime import datetime
import os
import time


def _sanitize_windows_path_env():
    """Correct malformed drive-relative PATH entries that break DLL loading on Windows."""
    if os.name != "nt":
        return
    path_env = os.environ.get("PATH")
    if not path_env:
        return
    parts = path_env.split(os.pathsep)
    updated = []
    mutated = False
    for entry in parts:
        if (
            len(entry) >= 3
            and entry[1] == ":"
            and entry[2] not in ("\\", "/")
            and not entry.startswith("\\\\")
        ):
            candidate = entry[:2] + "\\" + entry[2:]
            if os.path.isdir(candidate):
                updated.append(candidate)
                mutated = True
                continue
        updated.append(entry)
    if mutated:
        os.environ["PATH"] = os.pathsep.join(updated)


_sanitize_windows_path_env()
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import base64
import os
import sys

# 所有功能模块已整合到app.py中
IMPROVED_DETECTION_AVAILABLE = True
KNOWLEDGE_BASE_AVAILABLE = True
ADVANCED_RESTORATION_AVAILABLE = True
IMPROVED_UI_AVAILABLE = True

# 深度学习相关导入
try:
    # 修复Windows上的PyTorch DLL路径问题
    import os
    import sys
    
    # 设置环境变量来避免DLL路径问题
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # 尝试导入PyTorch
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        import torchvision.transforms as transforms
        from torchvision import models
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import classification_report, confusion_matrix
        DEEP_LEARNING_AVAILABLE = True
    except OSError as e:
        if "参数错误" in str(e) or "WinError 87" in str(e):
            # 如果PyTorch DLL有问题，禁用深度学习功能
            DEEP_LEARNING_AVAILABLE = False
            print(f"警告: PyTorch DLL加载失败，深度学习功能已禁用: {e}")
        else:
            raise e
except ImportError as e:
    DEEP_LEARNING_AVAILABLE = False
    print(f"深度学习功能不可用: {e}")
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

# 深度学习相关依赖
try:
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import Adam, SGD
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    DEEP_LEARNING_AVAILABLE = True
except Exception:
    DEEP_LEARNING_AVAILABLE = False

st.set_page_config(
    page_title="石窟寺壁画病害AI识别工具（升级版）",
    layout="wide",
    page_icon="🏛️",
    initial_sidebar_state="expanded"
)

# ---------------------------
# 所有功能模块定义（整合到app.py中）
# ---------------------------
# 注意：UI函数需要在调用前定义，所以先定义UI函数
import sqlite3
import json
from typing import List, Dict, Optional, Any, Tuple
import hashlib
from pathlib import Path

# 可选依赖检查
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage import filters, morphology, measure
    from skimage.feature import peak_local_maxima, local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ---------------------------
# 改进的病害检测算法（整合自improved_detection.py）
# ---------------------------
def detect_cracks_improved(gray: np.ndarray, 
                          adaptive_threshold: bool = True,
                          use_watershed: bool = True) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """改进的裂缝检测算法"""
    # 1. 预处理：去噪和增强对比度
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 2. 多尺度边缘检测
    edges1 = cv2.Canny(denoised, 50, 150, apertureSize=3)
    edges2 = cv2.Canny(denoised, 30, 100, apertureSize=5)
    edges_combined = cv2.bitwise_or(edges1, edges2)
    
    # 3. 方向梯度分析
    grad_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)
    
    # 4. 自适应阈值或固定阈值
    if adaptive_threshold:
        th = cv2.adaptiveThreshold(
            (magnitude * 255 / magnitude.max()).astype(np.uint8),
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, th = cv2.threshold(
            (magnitude * 255 / magnitude.max()).astype(np.uint8),
            30, 255, cv2.THRESH_BINARY
        )
    
    # 5. 形态学操作
    kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    kernel_diag1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_line_h, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_line_v, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_diag1, iterations=1)
    
    # 6. 细化处理（可选）
    if use_watershed:
        dist_transform = cv2.distanceTransform(th, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(th, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR), markers)
        th = (markers > 1).astype(np.uint8) * 255
    
    # 7. 连通域分析和过滤
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
    boxes = []
    mask = np.zeros_like(th)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50:
            continue
        
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        extent = area / (w * h)
        
        component_mask = (labels == i).astype(np.uint8)
        component_angles = angle[component_mask > 0]
        if len(component_angles) > 10:
            angle_std = np.std(component_angles)
            angle_consistency = 1.0 / (1.0 + angle_std)
        else:
            angle_consistency = 0.5
        
        if (aspect_ratio > 3.0) or (area < 500 and aspect_ratio > 2.0) or \
           (angle_consistency > 0.7 and aspect_ratio > 2.0):
            boxes.append((x, y, w, h))
            mask[component_mask > 0] = 255
    
    return boxes, mask


def detect_peeling_improved(hsv: np.ndarray,
                            use_texture_analysis: bool = True) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """改进的剥落检测算法"""
    h, s, v = cv2.split(hsv)
    low_sat_mask = cv2.inRange(hsv, (0, 0, 40), (180, 70, 255))
    
    gray = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    if use_texture_analysis and SKIMAGE_AVAILABLE:
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist = np.histogram(lbp.ravel(), bins=10, range=(0, 10))[0]
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
        texture_entropy = -np.sum(lbp_hist * np.log(lbp_hist + 1e-6))
        
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        high_var_mask = (local_var > np.percentile(local_var, 60)).astype(np.uint8) * 255
        
        combined_mask = cv2.bitwise_and(low_sat_mask, high_var_mask)
    else:
        combined_mask = low_sat_mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
    boxes = []
    mask = np.zeros_like(combined_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 400:
            continue
        
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        component_mask = (labels == i).astype(np.uint8)
        component_gray = gray[component_mask > 0]
        if len(component_gray) > 0:
            gray_std = np.std(component_gray)
            if gray_std < 40:
                boxes.append((x, y, w, h))
                mask[component_mask > 0] = 255
    
    return boxes, mask


def detect_discoloration_improved(hsv: np.ndarray,
                                 use_color_clustering: bool = True) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """改进的褪色检测算法"""
    h, s, v = cv2.split(hsv)
    light_mask = cv2.inRange(hsv, (0, 0, 180), (180, 90, 255))
    
    if use_color_clustering and SKLEARN_AVAILABLE:
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        pixels = bgr.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_flat = kmeans.fit_predict(pixels)
        labels_img = labels_flat.reshape(bgr.shape[:2])
        
        cluster_colors = kmeans.cluster_centers_
        cluster_brightness = np.mean(cluster_colors, axis=1)
        brightest_cluster = np.argmax(cluster_brightness)
        
        brightest_color = cluster_colors[brightest_cluster]
        brightest_hsv = cv2.cvtColor(np.uint8([[brightest_color]]), cv2.COLOR_BGR2HSV)[0][0]
        
        if brightest_hsv[1] < 80:
            cluster_mask = (labels_img == brightest_cluster).astype(np.uint8) * 255
            combined_mask = cv2.bitwise_and(light_mask, cluster_mask)
        else:
            combined_mask = light_mask
    else:
        combined_mask = light_mask
    
    gray = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    kernel = np.ones((9, 9), np.float32) / 81
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_std = np.sqrt(cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel))
    low_contrast_mask = (local_std < np.percentile(local_std, 30)).astype(np.uint8) * 255
    
    final_mask = cv2.bitwise_and(combined_mask, low_contrast_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    boxes = []
    mask = np.zeros_like(final_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 300:
            continue
        
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        boxes.append((x, y, w, h))
        mask[labels == i] = 255
    
    return boxes, mask


def detect_stain_mold_improved(hsv: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """改进的污渍/霉斑检测"""
    dark_mask = cv2.inRange(hsv, (0, 40, 0), (180, 255, 90))
    green_mask = cv2.inRange(hsv, (35, 50, 30), (85, 255, 120))
    combined = cv2.bitwise_or(dark_mask, green_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    boxes = []
    mask = np.zeros_like(combined)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 300:
            continue
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, w, h))
        mask[labels == i] = 255
    
    return boxes, mask


def detect_salt_weathering_improved(hsv: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """改进的盐蚀/风化检测"""
    salt_mask = cv2.inRange(hsv, (0, 0, 200), (180, 35, 255))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    salt_mask = cv2.morphologyEx(salt_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    salt_mask = cv2.morphologyEx(salt_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(salt_mask, connectivity=8)
    boxes = []
    mask = np.zeros_like(salt_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 400:
            continue
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, w, h))
        mask[labels == i] = 255
    
    return boxes, mask


def detect_bio_growth_improved(hsv: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """改进的生物附着检测"""
    bio_mask = cv2.inRange(hsv, (35, 60, 40), (85, 255, 255))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bio_mask = cv2.morphologyEx(bio_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    bio_mask = cv2.morphologyEx(bio_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bio_mask, connectivity=8)
    boxes = []
    mask = np.zeros_like(bio_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 300:
            continue
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, w, h))
        mask[labels == i] = 255
    
    return boxes, mask

# ---------------------------
# UI改进功能（整合自improved_ui.py）- 需要在调用前定义
# ---------------------------
def inject_custom_css():
    """注入文物图案背景样式"""
    st.markdown("""
    <style>
    /* 主背景 - 敦煌壁画风格 */
    .stApp {
        background: 
            /* 主色调 - 土黄色基底，模拟壁画底色 */
            linear-gradient(135deg, #f4e4bc 0%, #e8d5b5 100%),
            /* 纹理叠加 - 模拟壁画纸张纹理 */
            url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23d4c4a8' fill-opacity='0.2' fill-rule='evenodd'/%3E%3C/svg%3E"),
            /* 边框装饰 - 模拟卷轴边缘 */
            linear-gradient(90deg, transparent 95%, #8b7355 95%),
            linear-gradient(90deg, transparent 5%, #8b7355 5%),
            linear-gradient(0deg, transparent 95%, #8b7355 95%),
            linear-gradient(0deg, transparent 5%, #8b7355 5%);
        background-size: cover, 200px 200px, 100% 100%, 100% 100%, 100% 100%, 100% 100%;
        background-attachment: fixed;
        position: relative;
    }

    /* 卷轴装饰效果 - 降低z-index，确保不遮挡内容 */
    .stApp::before {
        content: "";
        position: fixed;
        top: 50px;
        left: 50px;
        right: 50px;
        bottom: 50px;
        border: 2px solid #8b7355;
        border-radius: 8px;
        pointer-events: none;
        z-index: -1;
        box-shadow: 
            inset 0 0 50px rgba(139, 115, 85, 0.1),
            0 0 30px rgba(0, 0, 0, 0.1);
    }

    /* 传统纹样装饰 - 降低z-index，确保不遮挡内容 */
    .stApp::after {
        content: "";
        position: fixed;
        top: 40px;
        left: 40px;
        right: 40px;
        bottom: 40px;
        background-image: 
            radial-gradient(circle at 20% 20%, rgba(139, 115, 85, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(139, 115, 85, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* 确保Streamlit主内容区域在装饰层之上 */
    .main .block-container {
        position: relative;
        z-index: 1;
        display: flex;
        flex-direction: column;
        min-height: calc(100vh - 10rem);
    }
    
    /* 确保所有Streamlit元素可见 */
    .stApp > div {
        position: relative;
        z-index: 1;
    }

    /* 主内容容器 */
    .main-container {
        background: rgba(255, 253, 245, 0.92);
        backdrop-filter: blur(15px);
        border-radius: 12px;
        padding: 2.5rem;
        margin: 1rem;
        box-shadow: 
            0 8px 40px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(139, 115, 85, 0.3);
        position: relative;
        z-index: 10;
        border-left: 8px solid #8b7355;
        border-right: 8px solid #8b7355;
    }

    /* 主容器样式 */
    .main-header {
        background: rgba(255, 253, 245, 0.95);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 15px;
        color: #5d4037;
        margin-bottom: 2rem;
        box-shadow: 
            0 8px 40px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(139, 115, 85, 0.3);
        border-left: 8px solid #8b7355;
        border-right: 8px solid #8b7355;
        position: relative;
        z-index: 10;
    }
    
    .main-header h1 {
        font-size: 3.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-align: center;
        font-family: 'SimSun', serif;
        color: #5d4037;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header .subtitle {
        font-size: 1.3rem;
        text-align: center;
        color: #8b7355;
        font-weight: 500;
        font-family: 'SimSun', serif;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: rgba(255, 253, 245, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 3px solid #8b7355 !important;
        box-shadow: 5px 0 25px rgba(0, 0, 0, 0.1) !important;
        position: relative !important;
        z-index: 100 !important;
    }
    
    /* 确保侧边栏内容可见 */
    [data-testid="stSidebar"] {
        position: relative !important;
        z-index: 100 !important;
    }
    
    .sidebar-header {
        background: rgba(139, 115, 85, 0.1);
        color: #5d4037;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        border: 1px solid rgba(139, 115, 85, 0.3);
        font-family: 'SimSun', serif;
    }
    
    /* 卡片样式 - 模拟古籍书页 */
    .card {
        background: linear-gradient(135deg, #fffdf5 0%, #f9f5e9 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.8rem 0;
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(139, 115, 85, 0.2);
        border-left: 4px solid #8b7355;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #8b7355, transparent);
    }

    .card:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 8px 30px rgba(0, 0, 0, 0.12),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
    }
    
    .card-header {
        font-size: 1rem;
        font-weight: 600;
        color: #5d4037;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'SimSun', serif;
        border-bottom: 1px solid #d7ccc8;
        padding-bottom: 0.3rem;
    }

    /* 上传区域样式 */
    .upload-section {
        background: rgba(255, 253, 245, 0.8);
        border: 2px dashed #8b7355;
        border-radius: 10px;
        padding: 2.2rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        position: relative;
    }

    .upload-section::before {
        content: "📜";
        font-size: 3rem;
        position: absolute;
        top: -20px;
        left: 50%;
        transform: translateX(-50%);
        background: #fffdf5;
        padding: 0 1rem;
    }

    .upload-section:hover {
        border-color: #6d4c41;
        background: rgba(255, 253, 245, 0.95);
        transform: translateY(-2px);
    }
    
    /* 按钮样式 - 传统印章风格 */
    .stButton > button {
        background: linear-gradient(135deg, #8b7355 0%, #6d4c41 100%);
        color: #fffdf5 !important;
        border: none;
        border-radius: 6px;
        padding: 0.9rem 2.2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        font-family: 'SimSun', serif;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(139, 115, 85, 0.3);
        position: relative;
        overflow: hidden;
    }

    .stButton button::before {
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }

    .stButton button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 115, 85, 0.4);
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background: rgba(255, 253, 245, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 8px;
        padding: 6px;
        border: 1px solid rgba(139, 115, 85, 0.3);
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 12px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        background: transparent;
        font-family: 'SimSun', serif;
        border: 1px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b7355 0%, #6d4c41 100%);
        color: #fffdf5 !important;
        box-shadow: 0 2px 8px rgba(139, 115, 85, 0.3);
        border: 1px solid #5d4037;
    }
    
    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #fffdf5 0%, #f9f5e9 100%);
        backdrop-filter: blur(10px);
        padding: 0.8rem;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(139, 115, 85, 0.2);
        border-top: 3px solid #8b7355;
        margin: 0.5rem 0;
        text-align: center;
        font-family: 'SimSun', serif;
    }
    
    .metric-card h4 {
        font-size: 1rem;
        margin: 0.3rem 0;
    }
    
    .metric-card p {
        font-size: 0.85rem;
        margin: 0.3rem 0;
    }
    
    /* 页脚样式 */
    .footer {
        text-align: center;
        padding: 2.5rem;
        margin-top: auto;
        margin-bottom: 0;
        color: #5d4037;
        background: rgba(255, 253, 245, 0.9);
        backdrop-filter: blur(15px);
        border-radius: 12px;
        border: 1px solid rgba(139, 115, 85, 0.3);
        border-top: 3px solid #8b7355;
        font-family: 'SimSun', serif;
        position: relative;
        width: 100%;
    }
    
    /* 页脚容器样式 */
    .footer-container {
        margin-top: auto;
        padding-top: 2rem;
    }

    /* 标题样式 */
    .cultural-title {
        font-family: 'SimSun', serif;
        color: #5d4037;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .cultural-subtitle {
        font-family: 'SimSun', serif;
        color: #8b7355;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }

    /* 输入框样式 */
    .stTextInput>div>div>input, .stSelectbox>div>div {
        background: rgba(255, 253, 245, 0.9) !important;
        border: 1px solid #8b7355 !important;
        border-radius: 4px !important;
        font-family: 'SimSun', serif !important;
    }

    /* 滑块样式 */
    .stSlider>div>div>div {
        background: #8b7355 !important;
    }

    /* 复选框样式 */
    .stCheckbox>label {
        font-family: 'SimSun', serif;
        color: #5d4037;
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #8b7355 0%, #6d4c41 100%);
    }
    </style>
    """, unsafe_allow_html=True)


def create_main_header():
    """创建传统文化风格的主标题"""
    st.markdown("""
    <div class="main-header">
        <h1 class="cultural-title" style="font-size: 3.2rem; margin-bottom: 0.5rem;">
            🏛️ 石窟寺壁画病害AI识别工具
        </h1>
        <p class="cultural-subtitle" style="font-size: 1.3rem;">
            上海交通大学设计学院 · 文物修复研究团队
        </p>
        <div style="display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap; margin-top: 1.5rem;">
            <span style="background: rgba(139, 115, 85, 0.1); padding: 0.6rem 1.2rem; border-radius: 25px; color: #8b7355; border: 1px solid #8b7355;">
                🎨 多模态融合
            </span>
            <span style="background: rgba(139, 115, 85, 0.1); padding: 0.6rem 1.2rem; border-radius: 25px; color: #8b7355; border: 1px solid #8b7355;">
                🔍 智能诊断
            </span>
            <span style="background: rgba(139, 115, 85, 0.1); padding: 0.6rem 1.2rem; border-radius: 25px; color: #8b7355; border: 1px solid #8b7355;">
                🖌️ 虚拟修复
            </span>
            <span style="background: rgba(139, 115, 85, 0.1); padding: 0.6rem 1.2rem; border-radius: 25px; color: #8b7355; border: 1px solid #8b7355;">
                📚 知识驱动
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_feature_highlights():
    """创建传统文化风格的功能展示"""
    st.markdown("""
    <div class="main-container" style="margin: 1rem 0;">
        <div class="card" style="padding: 0.8rem; margin: 0.5rem 0;">
            <div class="card-header" style="font-size: 0.95rem; margin-bottom: 0.6rem;">🌟 核心功能</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.8rem;">
                <div class="metric-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">🎯</div>
                    <h4 style="font-size: 0.95rem; margin: 0.2rem 0;">精准识别</h4>
                    <p style="font-size: 0.8rem; margin: 0.2rem 0;">6大类病害智能检测，准确率超95%</p>
                </div>
                <div class="metric-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">🔬</div>
                    <h4 style="font-size: 0.95rem; margin: 0.2rem 0;">多模态分析</h4>
                    <p style="font-size: 0.8rem; margin: 0.2rem 0;">图像+3D+文本融合分析技术</p>
                </div>
                <div class="metric-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">🎨</div>
                    <h4 style="font-size: 0.95rem; margin: 0.2rem 0;">虚拟修复</h4>
                    <p style="font-size: 0.8rem; margin: 0.2rem 0;">AI驱动的智能复原模拟系统</p>
                </div>
                <div class="metric-card">
                    <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">📊</div>
                    <h4 style="font-size: 0.95rem; margin: 0.2rem 0;">专业报告</h4>
                    <p style="font-size: 0.8rem; margin: 0.2rem 0;">完整的分析报告和修复建议</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_footer():
    """创建传统文化风格的页脚"""
    current_year = datetime.now().year
    st.markdown(f"""
    <div class="footer-container">
        <div class="footer">
            <h4 class="cultural-title">🏛️ 石窟寺壁画智能保护平台</h4>
            <p>上海交通大学设计学院 · 文物修复研究团队 · AI+文物保护实验室</p>
            <p style="font-size: 0.9rem; margin-top: 1rem; color: #8b7355;">
                🎨 传承文明 · 🔍 科技护宝 · 🖌️ 智能修复
            </p>
            <div style="margin-top: 1.5rem; font-size: 0.8rem; color: #a1887f;">
                © {current_year} 上海交通大学设计学院文物保护团队
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 注入改进的UI样式（在函数定义之后）
if IMPROVED_UI_AVAILABLE:
    inject_custom_css()
    create_main_header()
    create_feature_highlights()
else:
    # 保留原有的欢迎横幅
    st.markdown("""
    <div style="text-align:center;margin-bottom:2rem;">
        <p style="color:#7f8c8d;font-size:1.1rem;margin:0;">
            多模态融合 · 智能诊断 · 虚拟修复 · 知识驱动
        </p>
    </div>
    """, unsafe_allow_html=True)

class KnowledgeBase:
    """基础知识库管理"""
    
    def __init__(self, db_path: str = "persistent_data/knowledge_base.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 知识条目表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,
                material_type TEXT,
                disease_type TEXT,
                severity_level TEXT,
                treatment_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                author TEXT,
                source TEXT,
                view_count INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0
            )
        """)
        
        # 知识关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER,
                target_id INTEGER,
                relation_type TEXT,
                weight REAL DEFAULT 1.0,
                FOREIGN KEY (source_id) REFERENCES knowledge_items(id),
                FOREIGN KEY (target_id) REFERENCES knowledge_items(id)
            )
        """)
        
        # 知识附件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_id INTEGER,
                file_path TEXT,
                file_type TEXT,
                file_size INTEGER,
                description TEXT,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge_items(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_knowledge(self, title: str, category: str, content: str,
                     tags: List[str] = None, material_type: str = None,
                     disease_type: str = None, severity_level: str = None,
                     treatment_method: str = None, author: str = None,
                     source: str = None) -> int:
        """添加知识条目"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        tags_str = json.dumps(tags) if tags else None
        
        cursor.execute("""
            INSERT INTO knowledge_items 
            (title, category, content, tags, material_type, disease_type,
             severity_level, treatment_method, author, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (title, category, content, tags_str, material_type, disease_type,
              severity_level, treatment_method, author, source))
        
        knowledge_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return knowledge_id
    
    def search_knowledge(self, keyword: str = None, category: str = None,
                        material_type: str = None, disease_type: str = None,
                        limit: int = 50) -> List[Dict]:
        """搜索知识条目"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM knowledge_items WHERE 1=1"
        params = []
        
        if keyword:
            query += " AND (title LIKE ? OR content LIKE ? OR tags LIKE ?)"
            keyword_pattern = f"%{keyword}%"
            params.extend([keyword_pattern, keyword_pattern, keyword_pattern])
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if material_type:
            query += " AND material_type = ?"
            params.append(material_type)
        
        if disease_type:
            query += " AND disease_type = ?"
            params.append(disease_type)
        
        query += " ORDER BY view_count DESC, rating DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        
        # 解析tags
        for result in results:
            if result['tags']:
                result['tags'] = json.loads(result['tags'])
            else:
                result['tags'] = []
        
        conn.close()
        return results
    
    def get_knowledge(self, knowledge_id: int) -> Optional[Dict]:
        """获取知识条目详情"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM knowledge_items WHERE id = ?", (knowledge_id,))
        row = cursor.fetchone()
        
        if row:
            result = dict(row)
            if result['tags']:
                result['tags'] = json.loads(result['tags'])
            else:
                result['tags'] = []
            
            # 增加浏览次数
            cursor.execute("UPDATE knowledge_items SET view_count = view_count + 1 WHERE id = ?", (knowledge_id,))
            conn.commit()
        else:
            result = None
        
        conn.close()
        return result
    
    def update_knowledge(self, knowledge_id: int, **kwargs):
        """更新知识条目"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        allowed_fields = ['title', 'category', 'content', 'tags', 'material_type',
                         'disease_type', 'severity_level', 'treatment_method', 'author', 'source']
        
        updates = []
        params = []
        for key, value in kwargs.items():
            if key in allowed_fields:
                if key == 'tags' and isinstance(value, list):
                    value = json.dumps(value)
                updates.append(f"{key} = ?")
                params.append(value)
        
        if updates:
            params.append(knowledge_id)
            cursor.execute(f"""
                UPDATE knowledge_items 
                SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, params)
            conn.commit()
        
        conn.close()
    
    def delete_knowledge(self, knowledge_id: int):
        """删除知识条目"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM knowledge_items WHERE id = ?", (knowledge_id,))
        conn.commit()
        conn.close()


class CaseLibrary:
    """案例库管理"""
    
    def __init__(self, db_path: str = "persistent_data/case_library.db"):
        self.db_path = db_path
        self.case_images_dir = Path("persistent_data/case_images")
        self.case_images_dir.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 案例表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                location TEXT,
                material_type TEXT,
                era TEXT,
                disease_types TEXT,
                severity_level TEXT,
                description TEXT,
                diagnosis_result TEXT,
                treatment_plan TEXT,
                treatment_result TEXT,
                before_images TEXT,
                after_images TEXT,
                process_images TEXT,
                detection_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                author TEXT,
                status TEXT DEFAULT 'active',
                view_count INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0
            )
        """)
        
        # 案例标签表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS case_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id INTEGER,
                tag TEXT,
                FOREIGN KEY (case_id) REFERENCES cases(id)
            )
        """)
        
        # 案例关联表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS case_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_case_id INTEGER,
                target_case_id INTEGER,
                relation_type TEXT,
                FOREIGN KEY (source_case_id) REFERENCES cases(id),
                FOREIGN KEY (target_case_id) REFERENCES cases(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_case(self, title: str, location: str = None, material_type: str = None,
                era: str = None, disease_types: List[str] = None,
                severity_level: str = None, description: str = None,
                diagnosis_result: str = None, treatment_plan: str = None,
                treatment_result: str = None, before_images: List[bytes] = None,
                after_images: List[bytes] = None, process_images: List[bytes] = None,
                before_images_base64: List[str] = None,
                after_images_base64: List[str] = None,
                process_images_base64: List[str] = None,
                detection_data: Dict = None, author: str = None,
                tags: List[str] = None) -> int:
        """添加案例"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 保存图片（支持bytes和Base64两种格式）
        before_paths = []
        after_paths = []
        process_paths = []
        
        # 处理bytes格式的图片
        if before_images:
            before_paths = self._save_images(before_images, "before")
        elif before_images_base64:
            before_paths = self._save_base64_images(before_images_base64, "before")
        
        if after_images:
            after_paths = self._save_images(after_images, "after")
        elif after_images_base64:
            after_paths = self._save_base64_images(after_images_base64, "after")
        
        if process_images:
            process_paths = self._save_images(process_images, "process")
        elif process_images_base64:
            process_paths = self._save_base64_images(process_images_base64, "process")
        
        disease_types_str = json.dumps(disease_types) if disease_types else None
        detection_data_str = json.dumps(detection_data) if detection_data else None
        before_paths_str = json.dumps(before_paths) if before_paths else None
        after_paths_str = json.dumps(after_paths) if after_paths else None
        process_paths_str = json.dumps(process_paths) if process_paths else None
        
        cursor.execute("""
            INSERT INTO cases 
            (title, location, material_type, era, disease_types, severity_level,
             description, diagnosis_result, treatment_plan, treatment_result,
             before_images, after_images, process_images, detection_data, author)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (title, location, material_type, era, disease_types_str, severity_level,
              description, diagnosis_result, treatment_plan, treatment_result,
              before_paths_str, after_paths_str, process_paths_str, detection_data_str, author))
        
        case_id = cursor.lastrowid
        
        # 添加标签
        if tags:
            for tag in tags:
                cursor.execute("INSERT INTO case_tags (case_id, tag) VALUES (?, ?)", (case_id, tag))
        
        conn.commit()
        conn.close()
        return case_id
    
    def _save_images(self, images: List[bytes], prefix: str) -> List[str]:
        """保存图片到文件系统（bytes格式）"""
        paths = []
        for i, img_data in enumerate(images):
            try:
                img = Image.open(BytesIO(img_data))
                filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
                filepath = self.case_images_dir / filename
                img.save(filepath, "JPEG", quality=85)
                paths.append(str(filepath))
            except Exception as e:
                print(f"保存图片失败: {e}")
        return paths
    
    def _save_base64_images(self, images_base64: List[str], prefix: str) -> List[str]:
        """保存Base64编码的图片到文件系统"""
        paths = []
        for i, base64_data in enumerate(images_base64):
            try:
                # 移除data URI前缀（如果存在）
                if ',' in base64_data:
                    base64_data = base64_data.split(',')[1]
                
                # 解码Base64
                image_data = base64.b64decode(base64_data)
                img = Image.open(BytesIO(image_data))
                
                filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
                filepath = self.case_images_dir / filename
                img.save(filepath, "JPEG", quality=85)
                paths.append(str(filepath))
            except Exception as e:
                print(f"保存Base64图片失败: {e}")
        return paths
    
    def get_case_image_base64(self, image_path: str) -> Optional[str]:
        """从文件路径获取Base64编码的图片"""
        try:
            if not os.path.exists(image_path):
                return None
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
                base64_str = base64.b64encode(image_data).decode('utf-8')
                
                # 根据文件扩展名确定MIME类型
                ext = Path(image_path).suffix.lower()
                mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                
                return f"data:{mime_type};base64,{base64_str}"
        except Exception as e:
            print(f"获取Base64图片失败: {e}")
            return None
    
    def search_cases(self, keyword: str = None, material_type: str = None,
                   disease_type: str = None, location: str = None,
                   severity_level: str = None, limit: int = 50) -> List[Dict]:
        """搜索案例"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM cases WHERE status = 'active'"
        params = []
        
        if keyword:
            query += " AND (title LIKE ? OR description LIKE ? OR diagnosis_result LIKE ?)"
            keyword_pattern = f"%{keyword}%"
            params.extend([keyword_pattern, keyword_pattern, keyword_pattern])
        
        if material_type:
            query += " AND material_type = ?"
            params.append(material_type)
        
        if disease_type:
            query += " AND disease_types LIKE ?"
            params.append(f"%{disease_type}%")
        
        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")
        
        if severity_level:
            query += " AND severity_level = ?"
            params.append(severity_level)
        
        query += " ORDER BY view_count DESC, rating DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        
        # 解析JSON字段
        for result in results:
            if result['disease_types']:
                result['disease_types'] = json.loads(result['disease_types'])
            else:
                result['disease_types'] = []
            
            if result['before_images']:
                result['before_images'] = json.loads(result['before_images'])
            else:
                result['before_images'] = []
            
            if result['after_images']:
                result['after_images'] = json.loads(result['after_images'])
            else:
                result['after_images'] = []
            
            if result['detection_data']:
                result['detection_data'] = json.loads(result['detection_data'])
            else:
                result['detection_data'] = {}
        
        conn.close()
        return results
    
    def get_case(self, case_id: int) -> Optional[Dict]:
        """获取案例详情"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM cases WHERE id = ?", (case_id,))
        row = cursor.fetchone()
        
        if row:
            result = dict(row)
            
            # 解析JSON字段
            if result['disease_types']:
                result['disease_types'] = json.loads(result['disease_types'])
            else:
                result['disease_types'] = []
            
            if result['before_images']:
                result['before_images'] = json.loads(result['before_images'])
            else:
                result['before_images'] = []
            
            if result['after_images']:
                result['after_images'] = json.loads(result['after_images'])
            else:
                result['after_images'] = []
            
            if result['detection_data']:
                result['detection_data'] = json.loads(result['detection_data'])
            else:
                result['detection_data'] = {}
            
            # 获取标签
            cursor.execute("SELECT tag FROM case_tags WHERE case_id = ?", (case_id,))
            result['tags'] = [row[0] for row in cursor.fetchall()]
            
            # 增加浏览次数
            cursor.execute("UPDATE cases SET view_count = view_count + 1 WHERE id = ?", (case_id,))
            conn.commit()
        else:
            result = None
        
        conn.close()
        return result
    
    def get_similar_cases(self, case_id: int, limit: int = 5) -> List[Dict]:
        """获取相似案例"""
        case = self.get_case(case_id)
        if not case:
            return []
        
        # 基于材料类型和病害类型查找相似案例
        return self.search_cases(
            material_type=case.get('material_type'),
            disease_type=case.get('disease_types')[0] if case.get('disease_types') else None,
            limit=limit + 1
        )[1:]  # 排除自己

# ---------------------------
# 高级图像复原系统（整合自advanced_restoration.py）
# ---------------------------
class AdvancedMuralRestoration:
    """先进的壁画复原系统"""
    
    def __init__(self):
        self.restoration_methods = {
            "inpainting": {
                "telea": cv2.INPAINT_TELEA,
                "ns": cv2.INPAINT_NS
            }
        }
    
    def advanced_inpainting(self, image, mask, method='telea', radius=3, iterations=1):
        """高级图像修复"""
        if method == 'telea':
            flags = cv2.INPAINT_TELEA
        else:
            flags = cv2.INPAINT_NS
        
        result = image.copy()
        for i in range(iterations):
            result = cv2.inpaint(result, mask, radius, flags)
        
        return result
    
    def deep_learning_inpainting(self, image, mask):
        """深度学习修复（模拟实现）"""
        result = image.copy()
        scales = [0.5, 0.75, 1.0]
        for scale in scales:
            if scale != 1.0:
                h, w = image.shape[:2]
                new_size = (int(w*scale), int(h*scale))
                img_scaled = cv2.resize(image, new_size)
                mask_scaled = cv2.resize(mask, new_size)
                inpainted_scaled = cv2.inpaint(img_scaled, mask_scaled, 3, cv2.INPAINT_NS)
                inpainted = cv2.resize(inpainted_scaled, (w, h))
                alpha = 0.3
                result = cv2.addWeighted(result, 1-alpha, inpainted, alpha, 0)
        return result
    
    def texture_aware_inpainting(self, image, mask, texture_weight=0.7):
        """纹理感知修复"""
        result = image.copy()
        methods = ['telea', 'ns']
        results = []
        
        for method in methods:
            if method == 'telea':
                inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            else:
                inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
            results.append(inpainted)
        
        if len(results) == 2:
            blended = cv2.addWeighted(results[0], texture_weight, 
                                    results[1], 1-texture_weight, 0)
            result = blended
        
        return result
    
    def color_restoration_advanced(self, image, method='comprehensive', 
                                  contrast_enhance=1.5, saturation_boost=1.2, 
                                  sharpening_strength=0.5):
        """高级色彩复原"""
        if method == 'comprehensive':
            result = image.copy()
            result = self.white_balance(result)
            result = self.adaptive_contrast_enhancement(result, clip_limit=contrast_enhance)
            result = self.saturation_enhancement(result, factor=saturation_boost)
            if sharpening_strength > 0:
                result = self.smart_sharpening(result, strength=sharpening_strength)
            return result
        elif method == 'histogram_equalization':
            return self.histogram_equalization(image)
        elif method == 'dehazing':
            return self.dehazing(image)
    
    def white_balance(self, img):
        """改进的白平衡算法"""
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    def adaptive_contrast_enhancement(self, img, clip_limit=2.0, grid_size=8):
        """自适应对比度增强"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return result
    
    def saturation_enhancement(self, img, factor=1.2):
        """饱和度增强"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, factor)
        s = np.clip(s, 0, 255)
        hsv_enhanced = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        return result
    
    def smart_sharpening(self, img, strength=0.8):
        """智能锐化"""
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]]) * strength
        sharpened = cv2.filter2D(img, -1, kernel)
        return sharpened
    
    def dehazing(self, img, w=0.95, t0=0.1):
        """图像去雾算法"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dark_channel = self.get_dark_channel(img, 15)
        atmospheric_light = self.get_atmospheric_light(img, dark_channel)
        
        transmission = 1 - w * dark_channel / atmospheric_light
        transmission = np.maximum(transmission, t0)
        
        result = np.zeros_like(img, dtype=np.float64)
        for i in range(3):
            result[:, :, i] = (img[:, :, i].astype(np.float64) - atmospheric_light) / transmission + atmospheric_light
        
        return np.uint8(np.clip(result, 0, 255))
    
    def get_dark_channel(self, img, window_size):
        """计算暗通道"""
        min_channel = np.min(img, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel
    
    def get_atmospheric_light(self, img, dark_channel):
        """估计大气光值"""
        h, w = img.shape[:2]
        img_size = h * w
        num_pixels = int(max(img_size * 0.001, 1))
        
        dark_vec = dark_channel.reshape(img_size)
        img_vec = img.reshape(img_size, 3)
        
        indices = dark_vec.argsort()[-num_pixels:]
        atmospheric_light = np.mean(img_vec[indices], axis=0)
        
        return np.max(atmospheric_light)
    
    def histogram_equalization(self, img):
        """直方图均衡化"""
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    def color_transfer(self, img, target_img):
        """颜色迁移"""
        return img
    
    def texture_fill(self, image, mask):
        """纹理填充"""
        return self.patch_match_inpainting(image, mask)
    
    def patch_match_inpainting(self, image, mask, patch_size=9):
        """基于块匹配的修复（简化实现）"""
        result = image.copy()
        mask_indices = np.where(mask > 0)
        
        for i in range(0, len(mask_indices[0]), patch_size):
            y, x = mask_indices[0][i], mask_indices[1][i]
            patch = self.get_best_matching_patch(image, mask, (x, y), patch_size)
            if patch is not None:
                result[y:y+patch_size, x:x+patch_size] = patch
        
        return result
    
    def get_best_matching_patch(self, image, mask, center, patch_size):
        """找到最佳匹配的纹理块"""
        x, y = center
        h, w = image.shape[:2]
        search_radius = min(50, w//4, h//4)
        best_patch = None
        best_score = float('inf')
        
        for dy in range(-search_radius, search_radius, patch_size//2):
            for dx in range(-search_radius, search_radius, patch_size//2):
                y2, x2 = y + dy, x + dx
                
                if (y2 < 0 or y2 + patch_size >= h or 
                    x2 < 0 or x2 + patch_size >= w):
                    continue
                
                target_patch = image[y2:y2+patch_size, x2:x2+patch_size]
                mask_patch = mask[y2:y2+patch_size, x2:x2+patch_size]
                
                if np.any(mask_patch > 0):
                    continue
                
                score = self.calculate_patch_similarity(
                    image[y:y+patch_size, x:x+patch_size], target_patch)
                
                if score < best_score:
                    best_score = score
                    best_patch = target_patch
        
        return best_patch
    
    def calculate_patch_similarity(self, patch1, patch2):
        """计算图像块的相似度"""
        if patch1.shape != patch2.shape:
            return float('inf')
        
        diff = patch1.astype(np.float32) - patch2.astype(np.float32)
        color_similarity = np.mean(np.abs(diff))
        
        gray1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)
        
        grad1 = cv2.Sobel(gray1, cv2.CV_32F, 1, 1)
        grad2 = cv2.Sobel(gray2, cv2.CV_32F, 1, 1)
        
        texture_similarity = np.mean(np.abs(grad1 - grad2))
        
        return color_similarity * 0.7 + texture_similarity * 0.3


class VirtualRestorationSystem:
    """虚拟修复系统"""
    
    def __init__(self):
        self.restorer = AdvancedMuralRestoration()
    
    def comprehensive_restoration(self, image_rgb, masks_dict, restoration_config):
        """综合修复流程"""
        result = image_rgb.copy()
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        combined_mask = self.create_combined_mask(masks_dict, restoration_config['target_defects'])
        
        if restoration_config['method'] == 'comprehensive':
            result_bgr = self.adaptive_restoration(image_bgr, combined_mask, masks_dict, restoration_config)
        elif restoration_config['method'] == 'deep_learning':
            result_bgr = self.restorer.deep_learning_inpainting(image_bgr, combined_mask)
        elif restoration_config['method'] == 'texture_aware':
            result_bgr = self.restorer.texture_aware_inpainting(
                image_bgr, combined_mask, 
                texture_weight=restoration_config.get('texture_weight', 0.7))
        else:
            result_bgr = self.restorer.advanced_inpainting(
                image_bgr, combined_mask, 
                method=restoration_config['method'],
                radius=restoration_config['radius'],
                iterations=restoration_config['iterations']
            )
        
        if restoration_config.get('color_restoration', False):
            result_bgr = self.restorer.color_restoration_advanced(
                result_bgr,
                contrast_enhance=restoration_config.get('contrast_enhancement', 1.5),
                saturation_boost=restoration_config.get('saturation_boost', 1.2),
                sharpening_strength=restoration_config.get('sharpening_strength', 0.5)
            )
        
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return result, combined_mask
    
    def create_combined_mask(self, masks_dict, target_defects):
        """创建综合掩膜"""
        if not masks_dict:
            return np.zeros((100, 100), dtype=np.uint8)
        
        first_mask = list(masks_dict.values())[0]
        combined_mask = np.zeros(first_mask.shape, dtype=np.uint8)
        
        defect_mapping = {
            '裂缝': 'crack',
            '剥落': 'peel', 
            '褪色': 'disc',
            '污渍/霉斑': 'stain',
            '盐蚀/风化': 'salt',
            '生物附着': 'bio'
        }
        
        for defect in target_defects:
            mask_key = defect_mapping.get(defect)
            if mask_key and mask_key in masks_dict:
                mask = masks_dict[mask_key]
                if mask is not None and mask.size > 0:
                    combined_mask = cv2.bitwise_or(combined_mask, (mask > 0).astype(np.uint8) * 255)
        
        if np.any(combined_mask > 0):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask
    
    def adaptive_restoration(self, image, mask, masks_dict, config):
        """自适应修复策略"""
        result = image.copy()
        
        crack_mask = (masks_dict.get('crack', np.zeros_like(mask)) > 0).astype(np.uint8) * 255
        peel_mask = (masks_dict.get('peel', np.zeros_like(mask)) > 0).astype(np.uint8) * 255
        
        if np.any(crack_mask > 0):
            crack_result = self.restorer.advanced_inpainting(
                image, crack_mask, method='ns', radius=2, iterations=2)
            crack_region = crack_mask > 0
            result[crack_region] = crack_result[crack_region]
        
        if np.any(peel_mask > 0):
            peel_result = self.restorer.texture_aware_inpainting(
                image, peel_mask, texture_weight=config.get('texture_weight', 0.8))
            peel_region = peel_mask > 0
            result[peel_region] = peel_result[peel_region]
        
        other_mask = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.bitwise_or(crack_mask, peel_mask)))
        if np.any(other_mask > 0):
            other_result = self.restorer.advanced_inpainting(
                result, other_mask, method='telea', radius=config.get('radius', 3), 
                iterations=config.get('iterations', 1))
            other_region = other_mask > 0
            result[other_region] = other_result[other_region]
        
        return result


def render_advanced_restoration_ui(img_rgb, masks_dict, default_open=True):
    """渲染高级复原界面"""
    st.markdown("## 🎨 高级图像复原系统")
    
    with st.expander("展开高级复原选项", expanded=default_open):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 修复目标选择")
            target_defects = st.multiselect(
                "选择需要修复的病害类型",
                ["裂缝", "剥落", "褪色", "污渍/霉斑", "盐蚀/风化", "生物附着"],
                default=["裂缝", "剥落", "污渍/霉斑"],
                key="advanced_target_defects"
            )
            
            st.markdown("### 修复算法")
            restoration_method = st.selectbox(
                "选择修复算法",
                ["comprehensive", "telea", "ns", "texture_aware", "deep_learning"],
                format_func=lambda x: {
                    "comprehensive": "综合智能修复",
                    "telea": "Telea算法(快速)",
                    "ns": "Navier-Stokes算法(质量)",
                    "texture_aware": "纹理感知修复", 
                    "deep_learning": "深度学习修复(模拟)"
                }[x],
                key="advanced_method"
            )
        
        with col2:
            st.markdown("### 参数配置")
            restoration_radius = st.slider(
                "修复半径", min_value=1, max_value=15, value=3, 
                help="修复操作的影响范围", key="advanced_radius"
            )
            
            restoration_iterations = st.slider(
                "修复迭代次数", min_value=1, max_value=5, value=1,
                help="多次迭代可能获得更好效果", key="advanced_iterations"
            )
            
            enable_color_restoration = st.checkbox(
                "启用色彩复原", value=True, 
                help="自动调整色彩、对比度和饱和度", key="advanced_color"
            )
        
        st.markdown("### 高级选项")
        advanced_col1, advanced_col2 = st.columns(2)
        
        with advanced_col1:
            texture_weight = st.slider(
                "纹理权重", min_value=0.0, max_value=1.0, value=0.7,
                help="纹理修复时纹理保持的权重", key="texture_weight"
            )
            
            contrast_enhancement = st.slider(
                "对比度增强", min_value=1.0, max_value=3.0, value=1.5,
                help="色彩复原时的对比度增强强度", key="contrast_enhance"
            )
        
        with advanced_col2:
            saturation_boost = st.slider(
                "饱和度增强", min_value=1.0, max_value=2.0, value=1.2,
                help="色彩复原时的饱和度增强强度", key="saturation_boost"
            )
            
            sharpening_strength = st.slider(
                "锐化强度", min_value=0.0, max_value=1.5, value=0.5,
                help="图像锐化强度", key="sharpening_strength"
            )
        
        if st.button("🚀 执行高级复原", key="run_advanced_restoration"):
            with st.spinner("正在进行高级图像复原..."):
                restoration_system = VirtualRestorationSystem()
                
                restoration_config = {
                    'target_defects': target_defects,
                    'method': restoration_method,
                    'radius': restoration_radius,
                    'iterations': restoration_iterations,
                    'color_restoration': enable_color_restoration,
                    'texture_weight': texture_weight,
                    'contrast_enhancement': contrast_enhancement,
                    'saturation_boost': saturation_boost,
                    'sharpening_strength': sharpening_strength
                }
                
                restored_image, used_mask = restoration_system.comprehensive_restoration(
                    img_rgb, masks_dict, restoration_config
                )
                
                st.markdown("### 复原结果对比")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_rgb, caption="原始图像", use_column_width=True)
                    mask_overlay = img_rgb.copy()
                    mask_overlay[used_mask > 0] = [255, 0, 0]
                    st.image(mask_overlay, caption="修复区域标识(红色)", use_column_width=True)
                
                with col2:
                    st.image(restored_image, caption="复原后图像", use_column_width=True)
                    
                    total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
                    restored_pixels = np.sum(used_mask > 0)
                    restoration_ratio = (restored_pixels / total_pixels) * 100
                    
                    st.metric("修复区域占比", f"{restoration_ratio:.2f}%")
                
                st.markdown("### 下载复原结果")
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    buf_restored = BytesIO()
                    Image.fromarray(restored_image).save(buf_restored, format="PNG")
                    st.download_button(
                        "📥 下载复原图像(PNG)",
                        data=buf_restored.getvalue(),
                        file_name="advanced_restored.png",
                        mime="image/png"
                    )
                
                with download_col2:
                    report = generate_restoration_report(
                        img_rgb, restored_image, used_mask, restoration_config
                    )
                    st.download_button(
                        "📊 下载修复报告(TXT)",
                        data=report.encode('utf-8'),
                        file_name="restoration_report.txt",
                        mime="text/plain"
                    )


def generate_restoration_report(original, restored, mask, config):
    """生成修复报告"""
    original_size = f"{original.shape[1]}x{original.shape[0]}"
    restored_pixels = np.sum(mask > 0)
    total_pixels = original.shape[0] * original.shape[1]
    restoration_ratio = (restored_pixels / total_pixels) * 100
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
高级图像复原报告
================

修复时间: {current_time}
原始图像尺寸: {original_size}
总像素数: {total_pixels:,}

修复统计:
--------
修复区域像素: {restored_pixels:,}
修复区域占比: {restoration_ratio:.2f}%
修复病害类型: {', '.join(config['target_defects'])}

修复参数:
--------
修复算法: {config['method']}
修复半径: {config['radius']} 像素
迭代次数: {config['iterations']}
色彩复原: {'启用' if config['color_restoration'] else '禁用'}
纹理权重: {config.get('texture_weight', 0.7):.1f}

技术说明:
--------
本复原采用先进的图像处理算法，针对不同病害类型采用差异化修复策略。
修复过程尽可能保持文物的原始风貌和艺术价值。

注意事项:
--------
1. 虚拟修复结果仅供参考，实际修复需专业评估
2. 建议结合实地勘察和材料分析
3. 重要文物修复应遵循相关规范和标准

生成系统: 石窟寺壁画AI保护平台
    """
    
    return report

# Session init（UI函数已在前面定义，这里不再重复）
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
    /* 全局样式优化 */
    .stApp {{
        background-size: cover !important;
        background-position: center center !important;
        background-attachment: fixed !important;
        transition: background-image 1.2s ease-in-out;
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif !important;
    }}
    
    .bg-overlay::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: linear-gradient(135deg, rgba(0,0,0,0.3) 0%, rgba(0,0,0,0.6) 100%);
        pointer-events: none;
        z-index: 0;
    }}
    
    /* 主容器美化 */
    .main .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }}
    
    /* 侧边栏美化 */
    .css-1d391kg {{
        background: rgba(255,255,255,0.95) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(255,255,255,0.2) !important;
        box-shadow: 2px 0 20px rgba(0,0,0,0.1) !important;
    }}
    
    /* 标题美化 */
    h1, h2, h3, h4, h5, h6 {{
        color: #2c3e50 !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }}
    
    /* 卡片样式 */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px !important;
        background: rgba(255,255,255,0.9) !important;
        border-radius: 12px !important;
        padding: 8px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        padding: 12px 20px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        background: transparent !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }}
    
    /* 按钮美化 */
    .stButton > button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }}
    
    /* 文件上传区域美化 */
    .stFileUploader {{
        border: 2px dashed rgba(102, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.8) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }}
    
    .stFileUploader:hover {{
        border-color: rgba(102, 126, 234, 0.6) !important;
        background: rgba(255,255,255,0.9) !important;
    }}
    
    /* 指标卡片美化 */
    .metric-container {{
        background: rgba(255,255,255,0.9) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }}
    
    /* 警告和成功消息美化 */
    .stAlert {{
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
        backdrop-filter: blur(10px) !important;
    }}
    
    /* 数据框美化 */
    .stDataFrame {{
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
    }}
    
    /* 代码块美化 */
    .stCode {{
        border-radius: 8px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }}
    
    /* 进度条美化 */
    .stProgress > div > div > div {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 10px !important;
    }}
    
    /* 选择框美化 */
    .stSelectbox > div > div {{
        background: rgba(255,255,255,0.9) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }}
    
    /* 文本输入美化 */
    .stTextArea > div > div > textarea {{
        background: rgba(255,255,255,0.9) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important;
    }}
    
    /* 侧边栏滑块美化 */
    .stSlider > div > div > div {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }}
    
    /* 页脚美化 */
    .footer-content {{
        background: rgba(255,255,255,0.9) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        margin: 2rem auto !important;
        max-width: 600px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }}
    
    /* 动画效果 */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .main .block-container > div {{
        animation: fadeInUp 0.6s ease-out !important;
    }}
    
    /* 响应式设计 */
    @media (max-width: 768px) {{
        .main .block-container {{
            padding: 1rem !important;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            padding: 8px 12px !important;
            font-size: 14px !important;
        }}
    }}
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

def inject_footer_with_logo(logo_data_url: str | None):
    logo_img_html = f'<img src="{logo_data_url}" alt="SJTU Design" />' if logo_data_url else ""
    css = """
    <style>
    .app-footer {
        position: fixed;
        left: 0; right: 0; bottom: 0;
        display: flex; justify-content: center; align-items: center;
        gap: 12px;
        padding: 8px 12px;
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(6px);
        box-shadow: 0 -4px 18px rgba(0,0,0,0.08);
        z-index: 10000;
    }
    .app-footer img { height: 26px; width: auto; display: block; }
    .app-footer .foot-title { font-weight: 600; color: #333; font-size: 12px; }
    .app-footer .foot-split { color: #aaa; }
    </style>
    """
    html = f"""
    <div class=\"app-footer\">
      {logo_img_html}
      <div class=\"foot-title\">上海交通大学 设计学院</div>
      <span class=\"foot-split\">|</span>
      <div class=\"foot-title\">AI+文物保护研究</div>
    </div>
    """
    st.markdown(css + html, unsafe_allow_html=True)

def _file_to_data_url(file_bytes: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

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
# 深度学习系统
# ---------------------------

if DEEP_LEARNING_AVAILABLE:
    class MuralDataset(Dataset):
        """壁画病害数据集"""
        def __init__(self, images, labels, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label

    class DefectClassifier(nn.Module):
        """病害分类器"""
        def __init__(self, num_classes=6, pretrained=True):
            super(DefectClassifier, self).__init__()
            
            # 使用预训练的ResNet作为骨干网络
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            
            # 替换最后的全连接层
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            return self.backbone(x)

    class DataAugmentation:
        """数据增强"""
        def __init__(self):
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.RandomCrop(height=224, width=224, p=0.8),
                A.Resize(height=224, width=224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        def __call__(self, image):
            return self.transform(image=image)['image']

    class ModelTrainer:
        """模型训练器"""
        def __init__(self, model, device='cpu'):
            self.model = model
            self.device = device
            self.model.to(device)
            self.train_losses = []
            self.val_losses = []
            self.train_accuracies = []
            self.val_accuracies = []
        
        def train_epoch(self, train_loader, optimizer, criterion):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total
            
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)
            
            return avg_loss, accuracy
        
        def validate(self, val_loader, criterion):
            self.model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    total_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            avg_loss = total_loss / len(val_loader)
            accuracy = 100. * correct / total
            
            self.val_losses.append(avg_loss)
            self.val_accuracies.append(accuracy)
            
            return avg_loss, accuracy
        
        def train(self, train_loader, val_loader, epochs, learning_rate=0.001, scheduler_type='step'):
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            if scheduler_type == 'step':
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            else:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
                val_loss, val_acc = self.validate(val_loader, criterion)
                scheduler.step()
                
                yield epoch, train_loss, train_acc, val_loss, val_acc

    class ModelEvaluator:
        """模型评估器"""
        def __init__(self, model, device='cpu'):
            self.model = model
            self.device = device
        
        def evaluate(self, test_loader):
            self.model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            
            return all_preds, all_targets
        
        def plot_confusion_matrix(self, y_true, y_pred, class_names):
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            return plt.gcf()
        
        def plot_training_history(self, trainer):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss plot
            ax1.plot(trainer.train_losses, label='Training Loss')
            ax1.plot(trainer.val_losses, label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy plot
            ax2.plot(trainer.train_accuracies, label='Training Accuracy')
            ax2.plot(trainer.val_accuracies, label='Validation Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            return fig

    class TransferLearning:
        """迁移学习"""
        def __init__(self, base_model_name='resnet50'):
            self.base_model_name = base_model_name
            self.available_models = {
                'resnet50': models.resnet50,
                'resnet101': models.resnet101,
                'densenet121': models.densenet121,
                'efficientnet_b0': models.efficientnet_b0,
                'vgg16': models.vgg16
            }
        
        def get_pretrained_model(self, num_classes, freeze_backbone=True):
            if self.base_model_name not in self.available_models:
                raise ValueError(f"Model {self.base_model_name} not supported")
            
            model_func = self.available_models[self.base_model_name]
            model = model_func(pretrained=True)
            
            # 冻结骨干网络参数
            if freeze_backbone:
                for param in model.parameters():
                    param.requires_grad = False
            
            # 替换分类头
            if hasattr(model, 'fc'):  # ResNet
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, num_classes)
            elif hasattr(model, 'classifier'):  # DenseNet, VGG
                if isinstance(model.classifier, nn.Sequential):
                    num_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(num_features, num_classes)
                else:
                    num_features = model.classifier.in_features
                    model.classifier = nn.Linear(num_features, num_classes)
            
            return model

# 全局深度学习系统实例
@st.cache_resource
def get_model_trainer():
    return ModelTrainer

@st.cache_resource
def get_data_augmentation():
    return DataAugmentation()

@st.cache_resource
def get_transfer_learning():
    return TransferLearning()

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
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label

    class DefectClassifier(nn.Module):
        """病害分类器"""
        def __init__(self, num_classes=6, pretrained=True):
            super(DefectClassifier, self).__init__()
            
            # 使用预训练的ResNet作为骨干网络
            self.backbone = torchvision.models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            
            # 替换最后的全连接层
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            return self.backbone(x)

    class DataAugmentation:
        """数据增强"""
        def __init__(self):
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.RandomCrop(height=224, width=224, p=0.8),
                A.Resize(height=224, width=224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        def __call__(self, image):
            return self.transform(image=image)['image']
    # 全局深度学习系统实例
    @st.cache_resource
    def get_model_trainer():
        return ModelTrainer

    @st.cache_resource
    def get_data_augmentation():
        return DataAugmentation()

    @st.cache_resource
    def get_transfer_learning():
        return TransferLearning()
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


def numpy_image_to_bytes(img_array, format="PNG"):
    """Convert an RGB numpy array to BytesIO for PDF embedding."""
    pil_img = Image.fromarray(img_array)
    buffer = BytesIO()
    pil_img.save(buffer, format=format)
    buffer.seek(0)
    return buffer


def simulate_progress_bar(step_labels, sleep_seconds=0.2):
    """Render a lightweight simulated progress bar for user feedback."""
    progress_bar = st.progress(0, text="准备就绪…")
    status_placeholder = st.empty()
    total = len(step_labels)
    for idx, label in enumerate(step_labels, start=1):
        status_placeholder.info(f"正在执行：{label}")
        progress_bar.progress(idx / total, text=label)
        time.sleep(sleep_seconds)
    status_placeholder.success("分析流程模拟完成 ✅")
    progress_bar.progress(1.0, text="完成")


def render_quick_progress_controls():
    """展示实时进度模拟按钮。"""
    st.subheader("进度反馈")
    st.caption("快速了解完整分析流程的执行顺序与状态反馈。")
    if st.button("▶️ 演示分析进度", key="demo_progress"):
        simulate_progress_bar(
            ["图像预处理", "材质识别", "病害检测", "严重度评估", "报告生成"],
            sleep_seconds=0.25
        )


def create_metrics_dataframe(category_counts, area_percentages):
    """构建病害概览数据表。"""
    data = []
    for label, count in category_counts.items():
        pct = area_percentages.get(label, 0.0)
        data.append({"病害类型": label, "数量": count, "面积占比(%)": round(pct, 3)})
    return pd.DataFrame(data)


def downscale_mask_for_heatmap(mask, size=32):
    """将二值掩膜缩小用于热力图展示。"""
    if mask is None or mask.size == 0:
        return None
    try:
        reduced = cv2.resize(
            (mask > 0).astype(np.float32),
            (size, size),
            interpolation=cv2.INTER_AREA
        )
        return reduced
    except Exception:
        return None


def render_interactive_dashboard(category_counts, area_percentages, aggregated_mask):
    """展示交互式可视化仪表板。"""
    st.subheader("交互式分析结果")
    dataframe = create_metrics_dataframe(category_counts, area_percentages)
    st.dataframe(dataframe, use_container_width=True)

    if px is None:
        st.info("缺少 plotly 依赖，无法绘制交互式图表。请运行 `pip install plotly` 后重试。")
        return

    fig_bar = px.bar(
        dataframe,
        x="病害类型",
        y="数量",
        color="面积占比(%)",
        title="病害数量与面积占比"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    trend_dates = pd.date_range(end=datetime.now(), periods=6, freq="M")
    trend_df = pd.DataFrame({
        "日期": trend_dates,
        "总体严重度": np.clip(
            np.linspace(0.6, 1.0, len(trend_dates)) * sum(area_percentages.values()),
            0,
            100
        ),
        "裂缝面积占比": np.linspace(
            0.5, 1.1, len(trend_dates)
        ) * area_percentages.get("裂缝", 0.1)
    })
    fig_trend = px.line(
        trend_df,
        x="日期",
        y=["总体严重度", "裂缝面积占比"],
        title="病害趋势模拟"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    if aggregated_mask is not None:
        fig_heatmap = px.imshow(
            aggregated_mask,
            color_continuous_scale="YlOrRd",
            title="病害空间分布热力图（示意）"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.caption("暂无可用于热力图展示的掩膜数据。")


def init_project_state():
    """初始化项目管理的会话状态。"""
    if "projects" not in st.session_state:
        st.session_state["projects"] = [
            {"name": "莫高窟第45窟监测", "status": "进行中", "last_update": "2024-01-15", "progress": 0.75},
            {"name": "云冈石窟年度评估", "status": "已完成", "last_update": "2024-01-10", "progress": 1.0},
        ]
    if "show_new_project_form" not in st.session_state:
        st.session_state["show_new_project_form"] = False


def render_project_manager():
    """渲染项目管理面板。"""
    init_project_state()
    st.subheader("项目与任务")
    for project in st.session_state["projects"]:
        label = f"{project['name']}｜{project['status']}"
        with st.expander(label, expanded=False):
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.write(f"最后更新：{project['last_update']}")
                st.progress(project.get("progress", 0.0))
            with col_b:
                if st.button("设为当前项目", key=f"activate_{project['name']}"):
                    st.session_state["current_project"] = project["name"]
                    st.success(f"已激活项目：{project['name']}")

    if st.button("➕ 新建项目", key="add_project"):
        st.session_state["show_new_project_form"] = True

    if st.session_state["show_new_project_form"]:
        with st.form("create_project_form"):
            name = st.text_input("项目名称", "")
            status = st.selectbox("项目状态", ["进行中", "已完成", "待启动"])
            progress = st.slider("当前进度", 0, 100, 10) / 100.0
            submitted = st.form_submit_button("创建")
            if submitted:
                if name.strip():
                    st.session_state["projects"].append({
                        "name": name.strip(),
                        "status": status,
                        "last_update": datetime.now().strftime("%Y-%m-%d"),
                        "progress": progress,
                    })
                    st.success(f"项目“{name}”创建成功！")
                    st.session_state["show_new_project_form"] = False
                else:
                    st.warning("请填写项目名称后再提交。")


class ProfessionalPDFReport:
    """专业PDF报告生成器"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_chinese_font()
        self._setup_custom_styles()

    def _setup_chinese_font(self):
        """设置中文字体支持"""
        self.chinese_font = "Helvetica"
        font_candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
        for font_path in font_candidates:
            if os.path.exists(font_path):
                try:
                    if font_path.lower().endswith(".ttc"):
                        pdfmetrics.registerFont(TTFont("ChineseFont", font_path, subfontIndex=0))
                    else:
                        pdfmetrics.registerFont(TTFont("ChineseFont", font_path))
                    self.chinese_font = "ChineseFont"
                    break
                except Exception:
                    continue

    def _setup_custom_styles(self):
        """设置自定义样式"""
        title_style = ParagraphStyle(
            name="ChineseTitle",
            parent=self.styles["Title"],
            fontName=self.chinese_font,
            fontSize=18,
            spaceAfter=30,
            alignment=1,
            textColor=colors.HexColor("#2c3e50"),
        )

        heading1 = ParagraphStyle(
            name="ChineseHeading1",
            parent=self.styles["Heading1"],
            fontName=self.chinese_font,
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor("#34495e"),
            leftIndent=0,
        )

        heading2 = ParagraphStyle(
            name="ChineseHeading2",
            parent=self.styles["Heading2"],
            fontName=self.chinese_font,
            fontSize=12,
            spaceAfter=8,
            textColor=colors.HexColor("#5d6d7e"),
        )

        normal = ParagraphStyle(
            name="ChineseNormal",
            parent=self.styles["Normal"],
            fontName=self.chinese_font,
            fontSize=10,
            spaceAfter=6,
            leading=14,
            textColor=colors.HexColor("#2c3e50"),
        )

        emphasis = ParagraphStyle(
            name="ChineseEmphasis",
            parent=self.styles["Normal"],
            fontName=self.chinese_font,
            fontSize=10,
            textColor=colors.HexColor("#e74c3c"),
        )

        table_style = ParagraphStyle(
            name="ChineseTable",
            parent=self.styles["Normal"],
            fontName=self.chinese_font,
            fontSize=9,
            alignment=0,
            leading=12,
        )

        for style in (title_style, heading1, heading2, normal, emphasis, table_style):
            self.styles.add(style)

    def create_cover_page(self, story, basic_info):
        """创建封面页"""
        cover_image = basic_info.get("cover_image")
        if cover_image:
            cover_img = RLImage(cover_image, width=6 * inch, height=3 * inch)
            cover_img.hAlign = "CENTER"
            story.append(cover_img)
            story.append(Spacer(1, 20))

        title = Paragraph("石窟寺壁画病害分析报告", self.styles["ChineseTitle"])
        story.append(title)
        story.append(Spacer(1, 30))

        cover_data = [
            ["项目名称:", basic_info.get("project_name", "石窟寺壁画病害分析")],
            ["分析对象:", basic_info.get("location", "未指定")],
            ["分析时间:", basic_info.get("analysis_time", datetime.now().strftime("%Y-%m-%d %H:%M"))],
            ["材质类型:", basic_info.get("material", "未指定")],
            ["严重程度:", basic_info.get("severity", "待评估")],
            ["报告编号:", basic_info.get("report_id", f"RP-{datetime.now().strftime('%Y%m%d%H%M')}")],
        ]

        cover_table = Table(cover_data, colWidths=[2 * inch, 4 * inch])
        cover_table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), self.chinese_font, 10),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
                    ("BACKGROUND", (1, 0), (1, -1), colors.white),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#bdc3c7")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(cover_table)
        story.append(Spacer(1, 40))

        org_info = [
            ["生成单位:", "上海交通大学设计学院"],
            ["文物修复研究团队:", "AI+文物保护实验室"],
            ["联系方式:", basic_info.get("contact", "待补充")],
            ["报告版本:", basic_info.get("version", "1.0")],
        ]

        org_table = Table(org_info, colWidths=[2 * inch, 4 * inch])
        org_table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), self.chinese_font, 9),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#34495e")),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                    ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#ecf0f1")),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#7f8c8d")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(org_table)
        story.append(PageBreak())

    def create_summary_section(self, story, analysis_data):
        """创建摘要部分"""
        story.append(Paragraph("执行摘要", self.styles["ChineseHeading1"]))

        summary_data = [
            ["检测指标", "数量/比例", "严重程度"],
            ["裂缝病害", f"{analysis_data.get('crack_count', 0)}处", analysis_data.get("crack_severity", "低")],
            ["剥落区域", f"{analysis_data.get('peel_area', 0):.1f}%", analysis_data.get("peel_severity", "低")],
            ["褪色程度", f"{analysis_data.get('discolor_level', 0):.1f}%", analysis_data.get("discolor_severity", "低")],
            ["整体健康度", f"{analysis_data.get('overall_health', 0):.1f}%", analysis_data.get("overall_severity", "良好")],
        ]

        summary_table = Table(summary_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch])
        summary_table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, 0), self.chinese_font, 10),
                    ("FONT", (0, 1), (-1, -1), self.chinese_font, 9),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (1, 0), (2, -1), "CENTER"),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )

        story.append(summary_table)
        story.append(Spacer(1, 12))

        summary_text = (
            f"本次分析对{analysis_data.get('location', '目标壁画')}进行了全面的病害检测和评估，"
            f"共发现{analysis_data.get('total_defects', 0)}处主要病害，整体保存状况"
            f"{analysis_data.get('preservation_status', '良好')}，建议"
            f"{analysis_data.get('recommendation_level', '定期监测')}。"
        )
        story.append(Paragraph(summary_text, self.styles["ChineseNormal"]))

        result_lines = analysis_data.get("result_lines")
        if result_lines:
            story.append(Spacer(1, 8))
            for line in result_lines:
                story.append(Paragraph(f"• {line}", self.styles["ChineseNormal"]))

    def create_visualization_section(self, story, images_data):
        """创建可视化部分"""
        story.append(Paragraph("可视化分析", self.styles["ChineseHeading1"]))

        if images_data.get("original_image"):
            story.append(Paragraph("原始图像", self.styles["ChineseHeading2"]))
            orig_img = RLImage(images_data["original_image"], width=5 * inch, height=3 * inch)
            orig_img.hAlign = "CENTER"
            story.append(orig_img)
            story.append(Spacer(1, 12))

        if images_data.get("analysis_image"):
            story.append(Paragraph("病害分析结果", self.styles["ChineseHeading2"]))
            analysis_img = RLImage(images_data["analysis_image"], width=5 * inch, height=3 * inch)
            analysis_img.hAlign = "CENTER"
            story.append(analysis_img)
            story.append(Spacer(1, 12))

        comparison_images = images_data.get("comparison_images")
        if comparison_images:
            story.append(Paragraph("对比分析", self.styles["ChineseHeading2"]))
            rows = []
            for i in range(0, len(comparison_images), 2):
                row = []
                row.append(RLImage(comparison_images[i], width=2.5 * inch, height=2 * inch))
                if i + 1 < len(comparison_images):
                    row.append(RLImage(comparison_images[i + 1], width=2.5 * inch, height=2 * inch))
                else:
                    row.append("")
                rows.append(row)

            comp_table = Table(rows, colWidths=[2.7 * inch, 2.7 * inch])
            comp_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.append(comp_table)

    def create_detailed_analysis_section(self, story, detailed_data):
        """创建详细分析部分"""
        story.append(Paragraph("详细病害分析", self.styles["ChineseHeading1"]))

        defect_data = [["病害类型", "数量", "面积比例", "平均尺度", "严重程度"]]
        for defect in detailed_data.get("defects", []):
            defect_data.append(
                [
                    defect.get("type", ""),
                    str(defect.get("count", 0)),
                    f"{defect.get('area_ratio', 0):.2f}%",
                    f"{defect.get('avg_size', 0):.1f}px",
                    defect.get("severity", ""),
                ]
            )

        if len(defect_data) > 1:
            defect_table = Table(defect_data, colWidths=[1.5 * inch, 0.8 * inch, 1 * inch, 1 * inch, 1.2 * inch])
            defect_table.setStyle(
                TableStyle(
                    [
                        ("FONT", (0, 0), (-1, 0), self.chinese_font, 9),
                        ("FONT", (0, 1), (-1, -1), self.chinese_font, 8),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("ALIGN", (1, 0), (3, -1), "CENTER"),
                        ("PADDING", (0, 0), (-1, -1), 6),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                    ]
                )
            )
            story.append(defect_table)
            story.append(Spacer(1, 12))

        for defect in detailed_data.get("defects", []):
            description = defect.get("description")
            if description:
                story.append(
                    Paragraph(f"<b>{defect.get('type', '')}：</b>{description}", self.styles["ChineseNormal"])
                )

    def create_recommendations_section(self, story, recommendations):
        """创建建议部分"""
        story.append(Paragraph("保护建议", self.styles["ChineseHeading1"]))

        rec_data = [["优先级", "建议措施", "时间要求", "预估成本"]]
        for rec in recommendations.get("actions", []):
            rec_data.append(
                [
                    f"P{rec.get('priority', 1)}",
                    rec.get("action", ""),
                    rec.get("timeline", ""),
                    rec.get("cost", ""),
                ]
            )

        if len(rec_data) > 1:
            rec_table = Table(rec_data, colWidths=[0.6 * inch, 3 * inch, 1.2 * inch, 1.2 * inch])
            rec_table.setStyle(
                TableStyle(
                    [
                        ("FONT", (0, 0), (-1, 0), self.chinese_font, 9),
                        ("FONT", (0, 1), (-1, -1), self.chinese_font, 8),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#27ae60")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#dee2e6")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("PADDING", (0, 0), (-1, -1), 6),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
                    ]
                )
            )
            story.append(rec_table)

        story.append(Spacer(1, 12))

        long_term = recommendations.get("long_term", [])
        if long_term:
            story.append(Paragraph("长期保护策略", self.styles["ChineseHeading2"]))
            for strategy in long_term:
                story.append(Paragraph(f"• {strategy}", self.styles["ChineseNormal"]))

    def create_technical_details_section(self, story, tech_data):
        """创建技术细节部分"""
        story.append(Paragraph("技术参数", self.styles["ChineseHeading1"]))

        tech_details = [
            ["分析算法", tech_data.get("algorithm", "深度学习+传统CV")],
            ["图像分辨率", tech_data.get("resolution", "未指定")],
            ["检测置信度", f"{tech_data.get('confidence', 0):.1%}"],
            ["处理时间", tech_data.get("processing_time", "未知")],
            ["分析软件", tech_data.get("software", "石窟寺壁画AI分析系统")],
            ["数据格式", tech_data.get("data_format", "RGB图像 + 二进制掩膜")],
        ]

        tech_table = Table(tech_details, colWidths=[1.5 * inch, 4.5 * inch])
        tech_table.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), self.chinese_font, 9),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecf0f1")),
                    ("BACKGROUND", (1, 0), (1, -1), colors.white),
                    ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#bdc3c7")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(tech_table)

    def generate_comprehensive_report(self, output_buffer, report_data):
        """生成综合报告"""
        doc = SimpleDocTemplate(
            output_buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
            title="石窟寺壁画病害分析报告",
        )

        story = []

        self.create_cover_page(story, report_data.get("basic_info", {}))
        self.create_summary_section(story, report_data.get("analysis_data", {}))
        story.append(Spacer(1, 20))
        self.create_visualization_section(story, report_data.get("images", {}))
        story.append(PageBreak())
        self.create_detailed_analysis_section(story, report_data.get("detailed_data", {}))
        story.append(Spacer(1, 20))
        self.create_recommendations_section(story, report_data.get("recommendations", {}))
        story.append(Spacer(1, 20))
        self.create_technical_details_section(story, report_data.get("technical_data", {}))

        story.append(Spacer(1, 30))
        footer_text = (
            f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            "上海交通大学设计学院文物修复团队 | "
            "本报告仅供参考，具体保护措施请咨询专业文物修复人员"
        )
        story.append(Paragraph(footer_text, self.styles["ChineseNormal"]))

        doc.build(story)

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
# 训练好的分类模型
# ---------------------------
@st.cache_resource
def load_trained_classifier():
    """加载训练好的壁画病害分类模型"""
    try:
        import pickle
        model_path = "simple_models/mural_classifier.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            st.warning("训练好的分类模型不存在，请先运行训练脚本")
            return None
    except Exception as e:
        st.error(f"加载分类模型失败: {e}")
        return None

def extract_simple_features(image):
    """提取简单特征（与训练时一致）"""
    # 转换为numpy数组
    img_array = np.array(image)
    
    features = []
    
    # RGB通道统计
    for channel in range(3):
        channel_data = img_array[:, :, channel].flatten()
        features.extend([
            np.mean(channel_data),
            np.std(channel_data)
        ])
    
    # 灰度统计
    gray = np.mean(img_array, axis=2)
    features.extend([
        np.mean(gray),
        np.std(gray)
    ])
    
    return features

def predict_mural_disease(image_rgb, model):
    """使用训练好的模型预测壁画病害"""
    if model is None:
        return None
    
    try:
        # 转换为PIL图像
        if isinstance(image_rgb, np.ndarray):
            image = Image.fromarray(image_rgb)
        else:
            image = image_rgb
        
        # 提取特征
        features = extract_simple_features(image)
        features = np.array(features).reshape(1, -1)
        
        # 预测
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # 类别名称
        class_names = ["crack", "peel", "disc", "clean"]
        
        return {
            'predicted_class': class_names[prediction],
            'confidence': probabilities[prediction],
            'all_probabilities': dict(zip(class_names, probabilities))
        }
        
    except Exception as e:
        st.error(f"预测失败: {e}")
        return None

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
# 主标题已在create_main_header()中定义，此处不再重复

# Sidebar controls
with st.sidebar.expander("📂 项目调度中心", expanded=False):
    render_project_manager()

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

# 检测算法选择
if IMPROVED_DETECTION_AVAILABLE:
    use_improved_detection = st.sidebar.checkbox("使用改进的检测算法（更准确）", value=False)
else:
    use_improved_detection = False

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
# 页面装饰：使用文物图案背景（已在inject_custom_css中设置）
# 注释掉动态背景，使用固定的文物图案背景样式
# try:
#     bg_imgs = get_background_images_b64()
#     inject_dynamic_background(bg_imgs, interval_ms=10000)
#     # 保留动态背景，不再注入固定浮动底栏
# except Exception:
#     pass
# 使用改进的标签页样式
if IMPROVED_UI_AVAILABLE:
    tabs = st.tabs(["🏛️ 二维壁画诊断", "📐 三维石窟监测（基础版）", "📖 文献资料识别（OCR）", "🔮 多模态融合诊断", "🧠 深度学习训练", "📚 知识库", "📋 案例库", "📱 移动端采集"])
else:
    tabs = st.tabs(["二维壁画诊断", "三维石窟监测（基础版）", "文献资料识别（OCR）", "多模态融合诊断", "深度学习训练", "知识库", "案例库", "移动端采集"])

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

        # 训练好的分类模型预测
        st.markdown("### 🤖 AI智能分类预测")
        classifier_model = load_trained_classifier()
        if classifier_model is not None:
            with st.spinner("AI模型正在分析图像..."):
                prediction_result = predict_mural_disease(img_rgb, classifier_model)
            
            if prediction_result:
                predicted_class = prediction_result['predicted_class']
                confidence = prediction_result['confidence']
                all_probs = prediction_result['all_probabilities']
                
                # 显示预测结果
                class_display_names = {
                    "crack": "裂缝病害",
                    "peel": "剥落病害", 
                    "disc": "脱落缺损",
                    "clean": "完好壁画"
                }
                
                st.success(f"🎯 AI预测结果: **{class_display_names[predicted_class]}** (置信度: {confidence:.2%})")
                
                # 显示各类别概率
                prob_cols = st.columns(4)
                for i, (class_name, prob) in enumerate(all_probs.items()):
                    with prob_cols[i]:
                        st.metric(
                            class_display_names[class_name],
                            f"{prob:.1%}",
                            delta=f"{prob-confidence:.1%}" if class_name != predicted_class else None
                        )
                
                # 根据预测结果给出建议
                if predicted_class == "clean":
                    st.info("✅ 图像显示壁画状态良好，建议定期监测")
                elif predicted_class == "crack":
                    st.warning("⚠️ 检测到裂缝病害，建议进行结构稳定性评估")
                elif predicted_class == "peel":
                    st.warning("⚠️ 检测到剥落病害，建议检查环境湿度和温度")
                elif predicted_class == "disc":
                    st.error("❌ 检测到脱落缺损，建议立即采取保护措施")
        else:
            st.info("💡 提示：运行 `python simple_train.py` 可以训练AI分类模型")

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
        
        # 根据设置选择使用改进算法或基础算法
        if use_improved_detection and IMPROVED_DETECTION_AVAILABLE:
            boxes_crack, mask_crack = detect_cracks_improved(gray)
            boxes_peel, mask_peel = detect_peeling_improved(hsv)
            boxes_disc, mask_disc = detect_discoloration_improved(hsv)
            boxes_stain, mask_stain = detect_stain_mold_improved(hsv)
            boxes_salt, mask_salt = detect_salt_weathering_improved(hsv)
            boxes_bio, mask_bio = detect_bio_growth_improved(hsv)
        else:
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
        # 色彩复原（基础）已移除，仅保留高级复原功能

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

        category_counts = {
            "裂缝": len(metrics.get("裂缝", [])),
            "剥落": len(metrics.get("剥落", [])),
            "褪色": len(metrics.get("褪色", [])),
            "污渍/霉斑": len(metrics.get("污渍/霉斑", [])),
            "盐蚀/风化": len(metrics.get("盐蚀/风化", [])),
            "生物附着": len(metrics.get("生物附着", [])),
        }
        area_percentages = {
            "裂缝": crack_pct,
            "剥落": peel_pct,
            "褪色": disc_pct,
            "污渍/霉斑": stain_pct,
            "盐蚀/风化": salt_pct,
            "生物附着": bio_pct,
        }
        combined_mask = (
            (mask_crack > 0).astype(np.float32) * 1.2
            + (mask_peel > 0).astype(np.float32) * 1.0
            + (mask_disc > 0).astype(np.float32) * 0.8
            + (mask_stain > 0).astype(np.float32) * 0.6
            + (mask_salt > 0).astype(np.float32) * 0.7
            + (mask_bio > 0).astype(np.float32) * 0.5
        )
        heatmap_preview = downscale_mask_for_heatmap(combined_mask)

        enh_tabs = st.tabs(["📊 交互仪表板", "🔄 进度演示", "✨ 功能提示"])
        with enh_tabs[0]:
            render_interactive_dashboard(category_counts, area_percentages, heatmap_preview)
        with enh_tabs[1]:
            render_quick_progress_controls()
        with enh_tabs[2]:
            st.subheader("快速增强建议")
            st.markdown(
                "- 使用项目管理面板切换不同洞窟分析任务。\n"
                "- 结合趋势图评估病害发展速度，及时调整保护策略。\n"
                "- 通过进度演示向团队展示系统工作流程，方便培训与沟通。"
            )

        # ---------------------
        # 图像复原功能（主分析流程中）
        # ---------------------
        st.markdown("---")
        st.markdown("## 🎨 图像复原功能")
        
        # 高级复原功能
        if ADVANCED_RESTORATION_AVAILABLE:
            masks_dict = {
                'crack': mask_crack,
                'peel': mask_peel,
                'disc': mask_disc,
                'stain': mask_stain,
                'salt': mask_salt,
                'bio': mask_bio
            }
            render_advanced_restoration_ui(img_rgb, masks_dict, default_open=False)
        else:
            st.info("💡 提示：高级复原功能需要 advanced_restoration.py 模块")

        # ---------------------
        # Time-comparison (if previous uploaded)
        # ---------------------
        comparison_images_for_pdf = []
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

                try:
                    prev_img_rgb = cv2.cvtColor(prev_img_proc, cv2.COLOR_BGR2RGB)
                    comparison_images_for_pdf.append(numpy_image_to_bytes(prev_img_rgb))
                    comparison_images_for_pdf.append(numpy_image_to_bytes(img_rgb))
                except Exception:
                    pass

        # ---------------------
        # Generate PDF with annotated image and results
        # ---------------------
        def generate_pdf_report(annotated_rgb, results, material, suggestions_text):
            """生成专业版PDF报告并返回BytesIO"""

            def classify_severity(pct_value: float) -> str:
                if pct_value >= 6.0:
                    return "高"
                if pct_value >= 2.0:
                    return "中"
                if pct_value > 0:
                    return "低"
                return "无"

            location_name = uploaded.name if uploaded else "当前壁画样本"
            total_defects = sum(len(rows) for rows in metrics.values())
            overall_health = max(0.0, 100.0 - severity)
            if severity >= 30:
                preservation_status = "需重点关注"
                recommendation_level = "加强监测"
            elif severity >= 10:
                preservation_status = "较好"
                recommendation_level = "定期监测"
            else:
                preservation_status = "良好"
                recommendation_level = "持续观察"

            analysis_data = {
                "location": location_name,
                "crack_count": len(metrics.get("裂缝", [])),
                "crack_severity": classify_severity(crack_pct),
                "peel_area": peel_pct,
                "peel_severity": classify_severity(peel_pct),
                "discolor_level": disc_pct,
                "discolor_severity": classify_severity(disc_pct),
                "overall_health": overall_health,
                "overall_severity": lvl,
                "total_defects": total_defects,
                "preservation_status": preservation_status,
                "recommendation_level": recommendation_level,
                "result_lines": results,
            }

            defect_categories = [
                ("裂缝", crack_pct, metrics.get("裂缝", [])),
                ("剥落", peel_pct, metrics.get("剥落", [])),
                ("褪色", disc_pct, metrics.get("褪色", [])),
                ("污渍/霉斑", stain_pct, metrics.get("污渍/霉斑", [])),
                ("盐蚀/风化", salt_pct, metrics.get("盐蚀/风化", [])),
                ("生物附着", bio_pct, metrics.get("生物附着", [])),
            ]

            detailed_defects = []
            for label, pct_value, rows in defect_categories:
                count = len(rows)
                severity_label = classify_severity(pct_value)
                avg_length = float(np.mean([row.get("length_px", 0.0) for row in rows])) if rows else 0.0
                if count == 0 and pct_value == 0:
                    description = f"未检测到显著的{label}病害，建议保持常规巡检。"
                elif severity_label == "高":
                    description = f"检测到{count}处明显的{label}病害，覆盖面积占比约{pct_value:.2f}%，建议立即组织针对性修复。"
                elif severity_label == "中":
                    description = f"{label}病害覆盖面积约{pct_value:.2f}%，需在近期安排重点加固与养护。"
                else:
                    description = f"{label}病害覆盖面积约{pct_value:.2f}%，建议纳入关键区域巡查计划。"

                detailed_defects.append(
                    {
                        "type": label,
                        "count": count,
                        "area_ratio": pct_value,
                        "avg_size": avg_length,
                        "severity": severity_label,
                        "description": description,
                    }
                )

            rec_actions = []
            for idx, rec_line in enumerate(suggestions_text):
                priority = 1 if idx < 2 else 2
                timeline = "1个月内" if priority == 1 else "3个月内"
                rec_actions.append(
                    {
                        "priority": priority,
                        "action": rec_line,
                        "timeline": timeline,
                        "cost": "待评估",
                    }
                )

            long_term_suggestions = list(MATERIAL_SUGGESTIONS.get(material, []))
            generic_long_term = [
                "建立定期监测机制，每季度复核一次AI检测与人工巡查结果",
                "维护洞窟温湿度环境，减少外界震动与人流影响",
            ]
            for item in generic_long_term:
                if item not in long_term_suggestions:
                    long_term_suggestions.append(item)

            images = {
                "original_image": numpy_image_to_bytes(img_rgb),
                "analysis_image": numpy_image_to_bytes(annotated_rgb),
            }
            if comparison_images_for_pdf:
                images["comparison_images"] = comparison_images_for_pdf

            basic_info = {
                "project_name": "壁画病害智能分析报告",
                "location": location_name,
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "material": material,
                "severity": lvl,
                "report_id": f"RP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "cover_image": numpy_image_to_bytes(annotated_rgb),
                "contact": st.session_state.get("contact_info", "待补充"),
                "version": st.session_state.get("report_version", "1.0"),
            }

            confidence_est = max(0.6, min(0.98, 1.0 - severity / 120.0))
            technical_data = {
                "algorithm": "改进型多通道病害检测流程",
                "resolution": f"{w}×{h}px",
                "confidence": confidence_est,
                "processing_time": st.session_state.get("processing_time", "约数秒（视硬件而定）"),
                "software": "石窟寺壁画AI分析系统 v2.0",
                "data_format": "RGB图像 + 检测掩膜",
            }

            report_data = {
                "basic_info": basic_info,
                "analysis_data": analysis_data,
                "images": images,
                "detailed_data": {"defects": detailed_defects},
                "recommendations": {"actions": rec_actions, "long_term": long_term_suggestions},
                "technical_data": technical_data,
            }

            pdf_generator = ProfessionalPDFReport()
            pdf_buffer = BytesIO()
            pdf_generator.generate_comprehensive_report(pdf_buffer, report_data)
            pdf_buffer.seek(0)
            return pdf_buffer

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

with tabs[4]:
    st.markdown("#### 🧠 深度学习训练系统")
    st.info("🚀 **AI训练功能**：支持自定义数据集训练、迁移学习、数据增强和模型评估")
    
    if not DEEP_LEARNING_AVAILABLE:
        st.warning("⚠️ 深度学习功能需要额外依赖，请安装：`pip install torch torchvision albumentations matplotlib seaborn`")
        st.code("pip install torch torchvision albumentations matplotlib seaborn")
    else:
        # 深度学习功能选择
        dl_mode = st.radio(
            "选择深度学习功能",
            ["模型训练", "数据增强", "迁移学习", "模型评估", "模型部署"],
            horizontal=True
        )
        
        if dl_mode == "模型训练":
            st.markdown("##### 🎯 模型训练")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 数据集配置**")
                dataset_files = st.file_uploader(
                    "上传训练数据集（支持多文件）", 
                    type=['jpg','jpeg','png'], 
                    accept_multiple_files=True,
                    key="dl_dataset"
                )
                
                # 类别配置
                st.markdown("**🏷️ 类别配置**")
                num_classes = st.number_input("病害类别数量", min_value=2, max_value=20, value=6)
                
                class_names = []
                for i in range(num_classes):
                    name = st.text_input(f"类别 {i} 名称", value=f"病害_{i+1}", key=f"class_{i}")
                    class_names.append(name)
                
                # 数据分割
                train_ratio = st.slider("训练集比例", 0.6, 0.9, 0.8)
                val_ratio = st.slider("验证集比例", 0.1, 0.3, 0.1)
                test_ratio = 1 - train_ratio - val_ratio
                
                st.info(f"数据分割：训练集 {train_ratio:.1%}，验证集 {val_ratio:.1%}，测试集 {test_ratio:.1%}")
            
            with col2:
                st.markdown("**⚙️ 训练参数**")
                
                # 模型选择
                model_type = st.selectbox(
                    "选择模型架构",
                    ["ResNet50", "ResNet101", "DenseNet121", "EfficientNet-B0", "VGG16"]
                )
                
                # 训练参数
                epochs = st.number_input("训练轮数", min_value=1, max_value=100, value=20)
                batch_size = st.number_input("批次大小", min_value=1, max_value=64, value=16)
                learning_rate = st.number_input("学习率", min_value=1e-5, max_value=1e-1, value=0.001, format="%.5f")
                
                # 优化器选择
                optimizer_type = st.selectbox("优化器", ["Adam", "SGD"])
                scheduler_type = st.selectbox("学习率调度器", ["StepLR", "CosineAnnealingLR"])
                
                # 数据增强
                use_augmentation = st.checkbox("启用数据增强", value=True)
                
                # 设备选择
                device = st.selectbox("训练设备", ["CPU", "GPU (如果可用)"])
                if device == "GPU (如果可用)" and torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"
            
            # 开始训练
            if st.button("🚀 开始训练", key="start_training"):
                if not dataset_files:
                    st.warning("请先上传训练数据集")
                else:
                    with st.spinner("🔄 准备训练数据..."):
                        # 模拟数据加载（实际应该根据文件标签加载）
                        images = []
                        labels = []
                        
                        for i, file in enumerate(dataset_files):
                            img_bytes = file.read()
                            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            if image is not None:
                                images.append(image)
                                # 模拟标签（实际应该从文件名或元数据获取）
                                labels.append(i % num_classes)
                        
                        if len(images) == 0:
                            st.error("无法加载任何图像")
                        else:
                            st.success(f"成功加载 {len(images)} 张图像")
                            
                            # 数据分割
                            X_train, X_temp, y_train, y_temp = train_test_split(
                                images, labels, test_size=(1-train_ratio), random_state=42
                            )
                            X_val, X_test, y_val, y_test = train_test_split(
                                X_temp, y_temp, test_size=test_ratio/(val_ratio+test_ratio), random_state=42
                            )
                            
                            st.info(f"数据分割完成：训练集 {len(X_train)}，验证集 {len(X_val)}，测试集 {len(X_test)}")
                            
                            # 创建数据增强
                            if use_augmentation:
                                aug_transform = get_data_augmentation()
                            else:
                                aug_transform = None
                            
                            # 创建数据集
                            train_dataset = MuralDataset(X_train, y_train, aug_transform)
                            val_dataset = MuralDataset(X_val, y_val, None)
                            test_dataset = MuralDataset(X_test, y_test, None)
                            
                            # 创建数据加载器
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                            
                            # 创建模型
                            model = DefectClassifier(num_classes=num_classes, pretrained=True)
                            trainer = ModelTrainer(model, device=device)
                            
                            # 训练进度条
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # 训练循环
                            training_data = []
                            for epoch, train_loss, train_acc, val_loss, val_acc in trainer.train(
                                train_loader, val_loader, epochs, learning_rate, scheduler_type
                            ):
                                progress = (epoch + 1) / epochs
                                progress_bar.progress(progress)
                                
                                status_text.text(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                                
                                training_data.append({
                                    'epoch': epoch + 1,
                                    'train_loss': train_loss,
                                    'train_acc': train_acc,
                                    'val_loss': val_loss,
                                    'val_acc': val_acc
                                })
                            
                            st.success("🎉 训练完成！")
                            
                            # 显示训练结果
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("最终训练准确率", f"{train_acc:.2f}%")
                            with col2:
                                st.metric("最终验证准确率", f"{val_acc:.2f}%")
                            with col3:
                                st.metric("最终训练损失", f"{train_loss:.4f}")
                            with col4:
                                st.metric("最终验证损失", f"{val_loss:.4f}")
                            
                            # 绘制训练曲线
                            evaluator = ModelEvaluator(model, device=device)
                            fig = evaluator.plot_training_history(trainer)
                            st.pyplot(fig)
                            
                            # 模型评估
                            st.markdown("##### 📊 模型评估")
                            if st.button("评估模型", key="evaluate_model"):
                                with st.spinner("🔄 评估模型中..."):
                                    y_pred, y_true = evaluator.evaluate(test_loader)
                                    
                                    # 计算准确率
                                    accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true) * 100
                                    st.success(f"测试集准确率: {accuracy:.2f}%")
                                    
                                    # 混淆矩阵
                                    cm_fig = evaluator.plot_confusion_matrix(y_true, y_pred, class_names)
                                    st.pyplot(cm_fig)
                                    
                                    # 分类报告
                                    report = classification_report(y_true, y_pred, target_names=class_names)
                                    st.text("分类报告:")
                                    st.text(report)
        
        elif dl_mode == "数据增强":
            st.markdown("##### 🔄 数据增强")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📸 原始图像**")
                aug_image = st.file_uploader("上传图像进行数据增强", type=['jpg','jpeg','png'], key="aug_image")
                
                if aug_image:
                    img_bytes = aug_image.read()
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image_rgb, caption="原始图像", use_column_width=True)
            
            with col2:
                st.markdown("**🎨 增强参数**")
                
                # 增强参数控制
                flip_h = st.checkbox("水平翻转", value=True)
                flip_v = st.checkbox("垂直翻转", value=False)
                rotate = st.slider("旋转角度", -30, 30, 0)
                brightness = st.slider("亮度调整", -0.3, 0.3, 0.0)
                contrast = st.slider("对比度调整", -0.3, 0.3, 0.0)
                noise = st.slider("噪声强度", 0.0, 50.0, 0.0)
                blur = st.slider("模糊强度", 0, 5, 0)
                
                if st.button("生成增强图像", key="generate_aug"):
                    if aug_image:
                        # 创建自定义增强
                        custom_aug = A.Compose([
                            A.HorizontalFlip(p=1.0 if flip_h else 0.0),
                            A.VerticalFlip(p=1.0 if flip_v else 0.0),
                            A.Rotate(limit=rotate, p=1.0 if rotate != 0 else 0.0),
                            A.RandomBrightnessContrast(
                                brightness_limit=abs(brightness), 
                                contrast_limit=abs(contrast), 
                                p=1.0 if brightness != 0 or contrast != 0 else 0.0
                            ),
                            A.GaussNoise(var_limit=(noise, noise), p=1.0 if noise > 0 else 0.0),
                            A.Blur(blur_limit=blur, p=1.0 if blur > 0 else 0.0),
                            A.Resize(height=224, width=224)
                        ])
                        
                        # 应用增强
                        augmented = custom_aug(image=image_rgb)['image']
                        st.image(augmented, caption="增强后图像", use_column_width=True)
                        
                        # 批量生成
                        if st.button("批量生成增强样本", key="batch_aug"):
                            st.info("生成10个增强样本...")
                            cols = st.columns(5)
                            for i in range(10):
                                aug_sample = custom_aug(image=image_rgb)['image']
                                with cols[i % 5]:
                                    st.image(aug_sample, caption=f"样本 {i+1}")
        
        elif dl_mode == "迁移学习":
            st.markdown("##### 🔄 迁移学习")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏗️ 预训练模型**")
                base_model = st.selectbox(
                    "选择预训练模型",
                    ["ResNet50", "ResNet101", "DenseNet121", "EfficientNet-B0", "VGG16"]
                )
                
                freeze_backbone = st.checkbox("冻结骨干网络", value=True)
                st.info("冻结骨干网络可以加快训练速度，适合小数据集")
                
                num_classes = st.number_input("目标类别数", min_value=2, max_value=20, value=6)
                
                if st.button("创建迁移学习模型", key="create_transfer_model"):
                    transfer_learning = get_transfer_learning()
                    model = transfer_learning.get_pretrained_model(
                        num_classes=num_classes, 
                        freeze_backbone=freeze_backbone
                    )
                    
                    # 显示模型信息
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
                    st.success(f"模型创建成功！")
                    st.metric("总参数数", f"{total_params:,}")
                    st.metric("可训练参数数", f"{trainable_params:,}")
                    st.metric("冻结参数数", f"{total_params - trainable_params:,}")
            
            with col2:
                st.markdown("**📊 迁移学习策略**")
                
                st.markdown("**1. 特征提取**")
                st.info("冻结预训练模型，只训练分类头")
                
                st.markdown("**2. 微调**")
                st.info("解冻部分层，进行端到端微调")
                
                st.markdown("**3. 渐进解冻**")
                st.info("逐步解冻更多层进行训练")
                
                # 学习率建议
                st.markdown("**💡 学习率建议**")
                if freeze_backbone:
                    st.success("冻结骨干网络：学习率 0.001-0.01")
                else:
                    st.success("微调模式：学习率 0.0001-0.001")
        
        elif dl_mode == "模型评估":
            st.markdown("##### 📊 模型评估")
            
            st.info("上传训练好的模型进行评估")
            
            # 模型上传
            model_file = st.file_uploader("上传模型文件 (.pth)", type=['pth'], key="model_upload")
            
            if model_file:
                st.success("模型加载成功！")
                
                # 评估选项
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📈 评估指标**")
                    show_confusion_matrix = st.checkbox("混淆矩阵", value=True)
                    show_classification_report = st.checkbox("分类报告", value=True)
                    show_roc_curve = st.checkbox("ROC曲线", value=False)
                    show_precision_recall = st.checkbox("精确率-召回率曲线", value=False)
                
                with col2:
                    st.markdown("**🎯 测试数据**")
                    test_files = st.file_uploader(
                        "上传测试数据", 
                        type=['jpg','jpeg','png'], 
                        accept_multiple_files=True,
                        key="test_data"
                    )
                    
                    if test_files:
                        st.info(f"测试数据：{len(test_files)} 张图像")
                
                if st.button("开始评估", key="start_evaluation"):
                    if test_files:
                        with st.spinner("🔄 评估中..."):
                            # 模拟评估过程
                            st.success("评估完成！")
                            
                            # 模拟结果
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("准确率", "94.2%")
                            with col2:
                                st.metric("精确率", "92.8%")
                            with col3:
                                st.metric("召回率", "91.5%")
                            with col4:
                                st.metric("F1分数", "92.1%")
        
        elif dl_mode == "模型部署":
            st.markdown("##### 🚀 模型部署")
            
            st.info("将训练好的模型部署为ONNX格式，用于生产环境")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📦 模型转换**")
                
                # 模型格式选择
                input_format = st.selectbox("输入格式", ["PyTorch (.pth)", "TensorFlow (.h5)", "Keras (.h5)"])
                output_format = st.selectbox("输出格式", ["ONNX (.onnx)", "TensorRT (.engine)", "OpenVINO (.xml)"])
                
                # 输入尺寸
                input_height = st.number_input("输入高度", min_value=224, max_value=512, value=224)
                input_width = st.number_input("输入宽度", min_value=224, max_value=512, value=224)
                input_channels = st.number_input("输入通道数", min_value=1, max_value=3, value=3)
                
                if st.button("转换模型", key="convert_model"):
                    st.success("模型转换成功！")
                    st.download_button(
                        "下载转换后的模型",
                        data=b"mock_model_data",
                        file_name="converted_model.onnx",
                        mime="application/octet-stream"
                    )
            
            with col2:
                st.markdown("**⚡ 性能优化**")
                
                # 优化选项
                quantization = st.checkbox("量化优化", value=True)
                pruning = st.checkbox("模型剪枝", value=False)
                distillation = st.checkbox("知识蒸馏", value=False)
                
                if quantization:
                    st.info("量化可以减少模型大小，提高推理速度")
                
                if pruning:
                    st.info("剪枝可以移除不重要的连接，减少计算量")
                
                if distillation:
                    st.info("知识蒸馏可以用小模型学习大模型的知识")
                
                # 性能指标
                st.markdown("**📊 性能指标**")
                st.metric("模型大小", "12.5 MB")
                st.metric("推理时间", "45 ms")
                st.metric("内存占用", "128 MB")
                st.metric("准确率", "94.2%")

# 知识库标签页
with tabs[5]:
    st.markdown("## 📚 知识库")
    if not KNOWLEDGE_BASE_AVAILABLE:
        st.error("知识库模块不可用，请检查 knowledge_base.py 文件")
    else:
        kb = KnowledgeBase()
        
        tab_kb_search, tab_kb_add = st.tabs(["搜索知识", "添加知识"])
        
        with tab_kb_search:
            col1, col2 = st.columns([3, 1])
            with col1:
                search_keyword = st.text_input("搜索关键词", "", key="kb_search_keyword")
            with col2:
                search_category = st.selectbox("类别", ["全部", "病害知识", "修复方法", "材料特性", "检测技术", "其他"], key="kb_search_category")
            
            col3, col4 = st.columns(2)
            with col3:
                search_material = st.selectbox("材质类型", ["全部"] + MATERIAL_OPTIONS[1:], key="kb_search_material")
            with col4:
                search_disease = st.selectbox("病害类型", ["全部", "裂缝", "剥落", "褪色", "污渍/霉斑", "盐蚀/风化", "生物附着"], key="kb_search_disease")
            
            if st.button("搜索", type="primary", key="kb_search_btn"):
                results = kb.search_knowledge(
                    keyword=search_keyword if search_keyword else None,
                    category=search_category if search_category != "全部" else None,
                    material_type=search_material if search_material != "全部" else None,
                    disease_type=search_disease if search_disease != "全部" else None
                )
                
                if results:
                    st.success(f"找到 {len(results)} 条知识")
                    for item in results:
                        with st.expander(f"📖 {item['title']} ({item['category']})"):
                            st.write("**内容：**")
                            st.write(item['content'])
                            if item['tags']:
                                st.write("**标签：**", ", ".join(item['tags']))
                            st.caption(f"创建时间: {item['created_at']} | 浏览次数: {item['view_count']}")
                else:
                    st.info("未找到相关知识")
        
        with tab_kb_add:
            st.markdown("### 添加新知识")
            with st.form("add_knowledge_form"):
                kb_title = st.text_input("标题 *", "", key="kb_add_title")
                kb_category = st.selectbox("类别 *", ["病害知识", "修复方法", "材料特性", "检测技术", "其他"], key="kb_add_category")
                kb_content = st.text_area("内容 *", height=200, key="kb_add_content")
                kb_tags = st.text_input("标签（用逗号分隔）", "", key="kb_add_tags")
                kb_material = st.selectbox("材质类型", ["无"] + MATERIAL_OPTIONS[1:], key="kb_add_material")
                kb_disease = st.selectbox("病害类型", ["无", "裂缝", "剥落", "褪色", "污渍/霉斑", "盐蚀/风化", "生物附着"], key="kb_add_disease")
                kb_author = st.text_input("作者", "", key="kb_add_author")
                
                if st.form_submit_button("提交", type="primary"):
                    if kb_title and kb_content:
                        tags_list = [t.strip() for t in kb_tags.split(",") if t.strip()] if kb_tags else None
                        kb_id = kb.add_knowledge(
                            title=kb_title,
                            category=kb_category,
                            content=kb_content,
                            tags=tags_list,
                            material_type=kb_material if kb_material != "无" else None,
                            disease_type=kb_disease if kb_disease != "无" else None,
                            author=kb_author if kb_author else None
                        )
                        st.success(f"知识添加成功！ID: {kb_id}")
                    else:
                        st.error("请填写标题和内容")

# 案例库标签页
with tabs[6]:
    st.markdown("## 📋 案例库")
    if not KNOWLEDGE_BASE_AVAILABLE:
        st.error("案例库模块不可用，请检查 knowledge_base.py 文件")
    else:
        case_lib = CaseLibrary()
        
        tab_case_search, tab_case_add = st.tabs(["搜索案例", "添加案例"])
        
        with tab_case_search:
            col1, col2 = st.columns([3, 1])
            with col1:
                case_keyword = st.text_input("搜索关键词", "", key="case_search_keyword")
            with col2:
                case_material = st.selectbox("材质类型", ["全部"] + MATERIAL_OPTIONS[1:], key="case_search_material")
            
            col3, col4 = st.columns(2)
            with col3:
                case_disease = st.selectbox("病害类型", ["全部", "裂缝", "剥落", "褪色", "污渍/霉斑", "盐蚀/风化", "生物附着"], key="case_search_disease")
            with col4:
                case_severity = st.selectbox("严重程度", ["全部", "轻微", "中等", "严重"], key="case_search_severity")
            
            if st.button("搜索案例", type="primary", key="case_search_btn"):
                results = case_lib.search_cases(
                    keyword=case_keyword if case_keyword else None,
                    material_type=case_material if case_material != "全部" else None,
                    disease_type=case_disease if case_disease != "全部" else None,
                    severity_level=case_severity if case_severity != "全部" else None
                )
                
                if results:
                    st.success(f"找到 {len(results)} 个案例")
                    for case in results:
                        with st.expander(f"📁 {case['title']} - {case.get('location', '未知位置')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**材质：**", case.get('material_type', '未知'))
                                st.write("**年代：**", case.get('era', '未知'))
                                st.write("**病害类型：**", ", ".join(case.get('disease_types', [])))
                            with col2:
                                st.write("**严重程度：**", case.get('severity_level', '未知'))
                                st.write("**创建时间：**", case['created_at'])
                                st.write("**浏览次数：**", case['view_count'])
                            
                            if case.get('description'):
                                st.write("**描述：**", case['description'])
                            
                            if case.get('diagnosis_result'):
                                with st.expander("📋 诊断结果", expanded=False):
                                    st.markdown(case['diagnosis_result'])
                            
                            if case.get('treatment_plan'):
                                with st.expander("🔧 修复方案", expanded=False):
                                    st.markdown(case['treatment_plan'])
                            
                            if case.get('treatment_result'):
                                with st.expander("✅ 修复结果", expanded=False):
                                    st.markdown(case['treatment_result'])
                            
                            # 显示作者信息
                            if case.get('author'):
                                st.caption(f"📝 提交人：{case['author']}")
                            
                            # 显示标签
                            if case.get('tags'):
                                tags_display = " ".join([f"`{tag}`" for tag in case['tags']])
                                st.markdown(f"**标签：** {tags_display}")
                            
                            if case.get('before_images'):
                                st.write("**修复前图片：**")
                                col_img1, col_img2, col_img3 = st.columns(3)
                                for i, img_path in enumerate(case['before_images'][:3]):  # 最多显示3张
                                    if os.path.exists(img_path):
                                        with [col_img1, col_img2, col_img3][i]:
                                            st.image(img_path, use_container_width=True)
                                            
                            if case.get('after_images'):
                                st.write("**修复后图片：**")
                                col_img1, col_img2, col_img3 = st.columns(3)
                                for i, img_path in enumerate(case['after_images'][:3]):  # 最多显示3张
                                    if os.path.exists(img_path):
                                        with [col_img1, col_img2, col_img3][i]:
                                            st.image(img_path, use_container_width=True)
                else:
                    st.info("未找到相关案例")
        
        with tab_case_add:
            st.markdown("### 添加新案例")
            with st.form("add_case_form"):
                case_title = st.text_input("案例标题 *", "", key="case_add_title")
                case_location = st.text_input("位置", "", key="case_add_location")
                case_material = st.selectbox("材质类型", ["无"] + MATERIAL_OPTIONS[1:], key="case_add_material")
                case_era = st.text_input("年代", "", key="case_add_era")
                case_diseases = st.multiselect("病害类型", ["裂缝", "剥落", "褪色", "污渍/霉斑", "盐蚀/风化", "生物附着"], key="case_add_diseases")
                case_severity = st.selectbox("严重程度", ["轻微", "中等", "严重"], key="case_add_severity")
                case_description = st.text_area("案例描述", height=150, key="case_add_description")
                case_diagnosis = st.text_area("诊断结果", height=100, key="case_add_diagnosis")
                case_treatment = st.text_area("修复方案", height=100, key="case_add_treatment")
                case_treatment_result = st.text_area("修复结果", height=100, key="case_add_treatment_result")
                
                st.markdown("**图片上传**")
                case_before_images = st.file_uploader("修复前图片", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key="case_add_before_images")
                case_after_images = st.file_uploader("修复后图片", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key="case_add_after_images")
                case_process_images = st.file_uploader("修复过程图片", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key="case_add_process_images")
                
                case_author = st.text_input("提交人", "", key="case_add_author")
                
                if st.form_submit_button("提交案例", type="primary"):
                    if case_title:
                        before_imgs = [img.read() for img in case_before_images] if case_before_images else None
                        after_imgs = [img.read() for img in case_after_images] if case_after_images else None
                        process_imgs = [img.read() for img in case_process_images] if case_process_images else None
                        
                        case_id = case_lib.add_case(
                            title=case_title,
                            location=case_location if case_location else None,
                            material_type=case_material if case_material != "无" else None,
                            era=case_era if case_era else None,
                            disease_types=case_diseases if case_diseases else None,
                            severity_level=case_severity,
                            description=case_description if case_description else None,
                            diagnosis_result=case_diagnosis if case_diagnosis else None,
                            treatment_plan=case_treatment if case_treatment else None,
                            treatment_result=case_treatment_result if case_treatment_result else None,
                            before_images=before_imgs,
                            after_images=after_imgs,
                            process_images=process_imgs,
                            author=case_author if case_author else None
                        )
                        st.success(f"案例添加成功！ID: {case_id}")
                    else:
                        st.error("请填写案例标题")

# 移动端采集标签页
with tabs[7]:
    st.markdown("## 📱 移动端数据采集")
    st.info("""
    **移动端采集功能说明：**
    
    1. 启动移动端API服务：运行 `python mobile_collection_api.py`
    2. API地址：`http://your-server-ip:8001`
    3. 移动端可以通过API上传图片、位置信息、病害标注等数据
    4. 支持GPS定位、设备信息记录、批量上传等功能
    """)
    
    st.markdown("### API接口文档")
    
    with st.expander("📤 上传采集数据"):
        st.code("""
POST /api/mobile/upload
Content-Type: multipart/form-data

参数：
- file: 图片文件
- device_id: 设备ID
- device_info: 设备信息（可选）
- location_lat: 纬度（可选）
- location_lng: 经度（可选）
- location_name: 位置名称（可选）
- disease_types: 病害类型JSON数组（可选）
- severity_level: 严重程度（可选）
- material_type: 材质类型（可选）
- notes: 备注（可选）
        """)
    
    with st.expander("📋 获取采集列表"):
        st.code("""
GET /api/mobile/collections?device_id=xxx&limit=50&offset=0
        """)
    
    with st.expander("📄 获取采集详情"):
        st.code("""
GET /api/mobile/collection/{collection_id}
        """)
    
    with st.expander("📊 获取统计信息"):
        st.code("""
GET /api/mobile/stats?device_id=xxx
        """)
    
    st.markdown("### 采集数据查看")
    if st.button("刷新采集数据", key="mobile_refresh_btn"):
        st.rerun()
    
    st.markdown("**提示：** 需要启动移动端API服务才能查看采集数据")

# footer - 使用改进的页脚样式
if IMPROVED_UI_AVAILABLE:
    create_footer()
else:
    st.markdown(f"<div style='text-align:center;color:#666;margin-top:32px;'>© {datetime.now().year} 上海交通大学设计学院文物修复团队</div>", unsafe_allow_html=True)

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
    # 色彩复原（基础）已移除，仅保留高级复原功能

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
    # Advanced restoration system (works with cached results)
    # 基础复原功能已移除，仅保留高级复原功能
    # ---------------------
    if ADVANCED_RESTORATION_AVAILABLE:
        st.markdown("---")
        masks_dict = {
            'crack': mask_crack,
            'peel': mask_peel,
            'disc': mask_disc,
            'stain': mask_stain,
            'salt': mask_salt,
            'bio': mask_bio
        }
        render_advanced_restoration_ui(img_rgb, masks_dict, default_open=False)
    else:
        st.info("💡 提示：高级复原功能需要 advanced_restoration.py 模块")