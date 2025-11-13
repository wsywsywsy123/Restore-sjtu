#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
美观的石窟寺壁画病害AI识别工具界面组件
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO


def inject_custom_css():
    """注入自定义CSS样式"""
    st.markdown("""
    <style>
    /* 主容器样式 */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .main-header .subtitle {
        font-size: 1.2rem;
        text-align: center;
        opacity: 0.9;
        font-weight: 300;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* 卡片样式 */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    
    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* 上传区域样式 */
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .upload-section:hover {
        border-color: #764ba2;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8f9fa;
        border-radius: 12px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        background: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* 指标卡片 */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    /* 页脚样式 */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: #6c757d;
        border-top: 1px solid #e9ecef;
    }
    
    /* 图标样式 */
    .icon {
        font-size: 1.2rem;
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 成功消息样式 */
    .stSuccess {
        border-left: 4px solid #28a745;
    }
    
    /* 警告消息样式 */
    .stWarning {
        border-left: 4px solid #ffc107;
    }
    
    /* 错误消息样式 */
    .stError {
        border-left: 4px solid #dc3545;
    }
    
    /* 信息消息样式 */
    .stInfo {
        border-left: 4px solid #17a2b8;
    }
    </style>
    """, unsafe_allow_html=True)


def create_main_header():
    """创建主标题"""
    st.markdown("""
    <div class="main-header">
        <h1>🏛️ 石窟寺壁画病害AI识别工具</h1>
        <div class="subtitle">多模态融合 · 智能诊断 · 虚拟修复 · 知识驱动</div>
    </div>
    """, unsafe_allow_html=True)


def create_feature_highlights():
    """创建功能特性展示"""
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; 
                border-radius: 15px; 
                color: white; 
                margin: 2rem 0;'>
        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; text-align: center;'>
            <div>
                <h3>🎯 精准识别</h3>
                <p>6大类病害智能检测</p>
            </div>
            <div>
                <h3>🔬 多模态分析</h3>
                <p>图像+3D+文本融合</p>
            </div>
            <div>
                <h3>🎨 虚拟修复</h3>
                <p>AI驱动的复原模拟</p>
            </div>
            <div>
                <h3>📊 专业报告</h3>
                <p>完整的分析报告</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_footer():
    """创建页脚"""
    from datetime import datetime
    current_year = datetime.now().year
    st.markdown(f"""
    <div class="footer">
        <h4>🏛️ 石窟寺壁画智能保护平台</h4>
        <p>© {current_year} 上海交通大学设计学院文物修复团队 · AI+文物保护研究</p>
        <p style="font-size: 0.9rem; color: #868e96;">
            技术支持：深度学习 · 计算机视觉 · 多模态AI · 知识图谱
        </p>
    </div>
    """, unsafe_allow_html=True)


def create_enhanced_sidebar():
    """创建增强的侧边栏"""
    with st.sidebar:
        # 侧边栏标题
        st.markdown('<div class="sidebar-header">🎛️ 分析配置</div>', unsafe_allow_html=True)
        
        # 材质选择卡片
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">🏺 材质选择</div>', unsafe_allow_html=True)
        
        material = st.selectbox(
            "选择壁画材质",
            ["砂岩", "石灰岩", "灰泥地仗层", "木质基底", "未指定"],
            index=4,
            help="材质选择会影响病害评分和建议",
            key="enhanced_material"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            auto_material = st.checkbox("自动识别", help="启用智能材质识别", key="auto_material")
        with col2:
            use_improved_detection = st.checkbox("改进算法", help="使用改进的检测算法", key="use_improved_detection")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 算法配置卡片
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">⚡ 性能设置</div>', unsafe_allow_html=True)
        
        max_dim = st.slider(
            "最大处理分辨率", 
            min_value=512, 
            max_value=2048, 
            value=1024,
            step=64,
            help="较高的分辨率提供更精确的结果但需要更长的处理时间",
            key="enhanced_max_dim"
        )
        
        detection_threshold = st.slider(
            "检测敏感度",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="调整病害检测的敏感程度",
            key="enhanced_threshold"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 显示设置卡片
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">👁️ 显示设置</div>', unsafe_allow_html=True)
        
        display_mode = st.radio(
            "显示模式",
            ["智能叠加", "仅病害区域", "原始图像", "对比视图"],
            index=0,
            help="选择结果展示方式",
            key="enhanced_display_mode"
        )
        
        min_area = st.slider(
            "最小显示面积", 
            min_value=10, 
            max_value=1000, 
            value=100,
            step=10,
            help="过滤掉面积过小的检测结果",
            key="enhanced_min_area"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 高级选项
        with st.expander("🔧 高级选项"):
            use_multiscale = st.checkbox("启用多尺度分析", value=True, key="multiscale")
            use_gpu = st.checkbox("使用GPU加速", value=False, key="use_gpu")
            save_logs = st.checkbox("保存处理日志", value=True, key="save_logs")
        
        return {
            'material': material,
            'auto_material': auto_material,
            'use_improved_detection': use_improved_detection,
            'max_dim': max_dim,
            'detection_threshold': detection_threshold,
            'display_mode': display_mode,
            'min_area': min_area,
            'use_multiscale': use_multiscale,
            'use_gpu': use_gpu,
            'save_logs': save_logs
        }


def create_enhanced_upload_section():
    """创建增强的上传区域"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📤 图像上传</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('### 🖼️ 当前图像')
        st.markdown('**上传需要分析的壁画图像**')
        current_image = st.file_uploader(
            "选择文件",
            type=['jpg', 'jpeg', 'png'],
            key="enhanced_current_upload",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if current_image:
            st.success("✅ 图像上传成功！")
            # 显示预览
            image = Image.open(current_image)
            st.image(image, caption="当前图像预览", use_column_width=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('### 📊 历史图像')
        st.markdown('**上传历史图像用于对比分析**')
        historical_image = st.file_uploader(
            "选择文件", 
            type=['jpg', 'jpeg', 'png'],
            key="enhanced_historical_upload",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if historical_image:
            st.info("📅 历史图像已加载")
            # 显示预览
            image = Image.open(historical_image)
            st.image(image, caption="历史图像预览", use_column_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return current_image, historical_image


def create_enhanced_analysis_button():
    """创建增强的分析按钮"""
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 开始智能分析", use_container_width=True, key="enhanced_analyze_btn"):
            return True
    return False

