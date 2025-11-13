#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的病害检测算法
使用更先进的图像处理技术提高检测准确性
"""
import cv2
import numpy as np
from typing import Tuple, List

# 可选依赖
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

def detect_cracks_improved(gray: np.ndarray, 
                          adaptive_threshold: bool = True,
                          use_watershed: bool = True) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    改进的裂缝检测算法
    
    改进点：
    1. 自适应阈值处理
    2. 多尺度边缘检测
    3. 形态学细化
    4. 连通域分析
    5. 方向一致性验证
    """
    # 1. 预处理：去噪和增强对比度
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 2. 多尺度边缘检测
    # 使用Canny边缘检测的改进版本
    edges1 = cv2.Canny(denoised, 50, 150, apertureSize=3)
    edges2 = cv2.Canny(denoised, 30, 100, apertureSize=5)
    edges_combined = cv2.bitwise_or(edges1, edges2)
    
    # 3. 方向梯度分析（裂缝通常是线性的）
    grad_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)
    
    # 4. 自适应阈值或固定阈值
    if adaptive_threshold:
        # 使用局部自适应阈值
        th = cv2.adaptiveThreshold(
            (magnitude * 255 / magnitude.max()).astype(np.uint8),
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, th = cv2.threshold(
            (magnitude * 255 / magnitude.max()).astype(np.uint8),
            30, 255, cv2.THRESH_BINARY
        )
    
    # 5. 形态学操作：连接断开的裂缝
    kernel_line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    kernel_diag1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_line_h, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_line_v, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_diag1, iterations=1)
    
    # 6. 细化处理（可选）
    if use_watershed:
        # 使用距离变换和分水岭算法
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
        if area < 50:  # 最小面积阈值
            continue
        
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # 计算细长比（裂缝通常是细长的）
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        extent = area / (w * h)  # 填充度
        
        # 方向一致性检查
        component_mask = (labels == i).astype(np.uint8)
        component_angles = angle[component_mask > 0]
        if len(component_angles) > 10:
            angle_std = np.std(component_angles)
            angle_consistency = 1.0 / (1.0 + angle_std)  # 方向一致性
        else:
            angle_consistency = 0.5
        
        # 综合判断：细长比高 或 面积小但细长 或 方向一致
        if (aspect_ratio > 3.0) or (area < 500 and aspect_ratio > 2.0) or \
           (angle_consistency > 0.7 and aspect_ratio > 2.0):
            boxes.append((x, y, w, h))
            mask[component_mask > 0] = 255
    
    return boxes, mask


def detect_peeling_improved(hsv: np.ndarray,
                            use_texture_analysis: bool = True) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    改进的剥落检测算法
    
    改进点：
    1. 纹理分析
    2. 多特征融合
    3. 区域一致性检查
    """
    h, s, v = cv2.split(hsv)
    
    # 1. 基于HSV的初步检测
    # 剥落区域通常：低饱和度、中等亮度、颜色偏灰白
    low_sat_mask = cv2.inRange(hsv, (0, 0, 40), (180, 70, 255))
    
    # 2. 纹理分析（剥落区域纹理通常不均匀）
    gray = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    if use_texture_analysis and SKIMAGE_AVAILABLE:
        # 使用局部二值模式（LBP）检测纹理
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist = np.histogram(lbp.ravel(), bins=10, range=(0, 10))[0]
        lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
        texture_entropy = -np.sum(lbp_hist * np.log(lbp_hist + 1e-6))
        
        # 计算局部方差（剥落区域方差通常较高）
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        high_var_mask = (local_var > np.percentile(local_var, 60)).astype(np.uint8) * 255
        
        # 融合低饱和度和高方差区域
        combined_mask = cv2.bitwise_and(low_sat_mask, high_var_mask)
    else:
        combined_mask = low_sat_mask
    
    # 3. 形态学操作：去除噪声，连接区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
    boxes = []
    mask = np.zeros_like(combined_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 400:  # 最小面积阈值
            continue
        
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        # 检查区域一致性（剥落区域内部应该相对均匀）
        component_mask = (labels == i).astype(np.uint8)
        component_gray = gray[component_mask > 0]
        if len(component_gray) > 0:
            gray_std = np.std(component_gray)
            # 如果标准差太大，可能是误检
            if gray_std < 40:  # 阈值可调
                boxes.append((x, y, w, h))
                mask[component_mask > 0] = 255
    
    return boxes, mask


def detect_discoloration_improved(hsv: np.ndarray,
                                 use_color_clustering: bool = True) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    改进的褪色检测算法
    
    改进点：
    1. 颜色聚类分析
    2. 区域对比度分析
    3. 多尺度检测
    """
    h, s, v = cv2.split(hsv)
    
    # 1. 基于HSV的初步检测
    # 褪色区域：高亮度、低到中等饱和度
    light_mask = cv2.inRange(hsv, (0, 0, 180), (180, 90, 255))
    
    if use_color_clustering and SKLEARN_AVAILABLE:
        # 2. 颜色聚类：褪色区域颜色应该与周围不同
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 使用K-means聚类找到主要颜色
        pixels = bgr.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_flat = kmeans.fit_predict(pixels)
        labels_img = labels_flat.reshape(bgr.shape[:2])
        
        # 找到最亮的聚类（可能是褪色区域）
        cluster_colors = kmeans.cluster_centers_
        cluster_brightness = np.mean(cluster_colors, axis=1)
        brightest_cluster = np.argmax(cluster_brightness)
        
        # 如果最亮聚类的饱和度也较低，则认为是褪色
        brightest_color = cluster_colors[brightest_cluster]
        brightest_hsv = cv2.cvtColor(np.uint8([[brightest_color]]), cv2.COLOR_BGR2HSV)[0][0]
        
        if brightest_hsv[1] < 80:  # 低饱和度
            cluster_mask = (labels_img == brightest_cluster).astype(np.uint8) * 255
            combined_mask = cv2.bitwise_and(light_mask, cluster_mask)
        else:
            combined_mask = light_mask
    else:
        combined_mask = light_mask
    
    # 3. 对比度分析：褪色区域与周围对比度应该较低
    gray = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    
    # 计算局部对比度
    kernel = np.ones((9, 9), np.float32) / 81
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_std = np.sqrt(cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel))
    
    # 低对比度区域（可能是褪色）
    low_contrast_mask = (local_std < np.percentile(local_std, 30)).astype(np.uint8) * 255
    
    # 融合
    final_mask = cv2.bitwise_and(combined_mask, low_contrast_mask)
    
    # 4. 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 5. 连通域分析
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
    # 污渍通常：暗色、中等饱和度、可能有特定色调（绿色、黑色等）
    
    # 1. 暗色区域
    dark_mask = cv2.inRange(hsv, (0, 40, 0), (180, 255, 90))
    
    # 2. 绿色调（霉斑）
    green_mask = cv2.inRange(hsv, (35, 50, 30), (85, 255, 120))
    
    # 3. 融合
    combined = cv2.bitwise_or(dark_mask, green_mask)
    
    # 4. 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 5. 连通域分析
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
    # 盐析通常：极高亮度、极低饱和度、白色
    
    # 1. 高亮度低饱和度区域
    salt_mask = cv2.inRange(hsv, (0, 0, 200), (180, 35, 255))
    
    # 2. 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    salt_mask = cv2.morphologyEx(salt_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    salt_mask = cv2.morphologyEx(salt_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. 连通域分析
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
    # 生物附着通常：绿色调、高饱和度
    
    # 1. 绿色调高饱和度区域
    bio_mask = cv2.inRange(hsv, (35, 60, 40), (85, 255, 255))
    
    # 2. 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bio_mask = cv2.morphologyEx(bio_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    bio_mask = cv2.morphologyEx(bio_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. 连通域分析
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

