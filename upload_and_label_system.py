#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片上传和标注系统
用于收集用户上传的壁画图片并进行标注
"""

import streamlit as st
import os
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
import json
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageUploadSystem:
    def __init__(self, upload_dir="user_uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.categories = {
            "crack": "裂缝病害",
            "peel": "剥落病害", 
            "disc": "脱落缺损",
            "discoloration": "变色病害",
            "stain_mold": "污渍霉斑",
            "salt_weathering": "盐蚀风化",
            "bio_growth": "生物附着",
            "clean": "完好壁画"
        }
        
        for category in self.categories.keys():
            (self.upload_dir / category).mkdir(exist_ok=True)
        
        # 标注记录文件
        self.annotation_file = self.upload_dir / "annotations.json"
        self.load_annotations()
    
    def load_annotations(self):
        """加载标注记录"""
        if self.annotation_file.exists():
            with open(self.annotation_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {}
    
    def save_annotations(self):
        """保存标注记录"""
        with open(self.annotation_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
    
    def upload_image(self, uploaded_file, category, description=""):
        """上传并保存图片"""
        try:
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{category}_{timestamp}_{uploaded_file.name}"
            
            # 保存到对应类别目录
            category_dir = self.upload_dir / category
            file_path = category_dir / filename
            
            # 保存图片
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 记录标注信息
            annotation_id = f"{category}_{timestamp}"
            self.annotations[annotation_id] = {
                "filename": filename,
                "category": category,
                "description": description,
                "upload_time": timestamp,
                "file_path": str(file_path.relative_to(self.upload_dir))
            }
            
            self.save_annotations()
            logger.info(f"图片上传成功: {filename} -> {category}")
            return True, filename
            
        except Exception as e:
            logger.error(f"图片上传失败: {e}")
            return False, str(e)
    
    def get_statistics(self):
        """获取上传统计"""
        stats = {}
        total = 0
        
        for category in self.categories.keys():
            category_dir = self.upload_dir / category
            count = len(list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png")) + list(category_dir.glob("*.jpeg")))
            stats[category] = count
            total += count
        
        stats['total'] = total
        return stats
    
    def get_annotation_dataframe(self):
        """获取标注数据框"""
        if not self.annotations:
            return pd.DataFrame()
        
        data = []
        for ann_id, ann_data in self.annotations.items():
            data.append({
                "ID": ann_id,
                "文件名": ann_data["filename"],
                "类别": self.categories.get(ann_data["category"], ann_data["category"]),
                "描述": ann_data.get("description", ""),
                "上传时间": ann_data["upload_time"]
            })
        
        return pd.DataFrame(data)
    
    def prepare_training_data(self, output_dir="expanded_training_dataset"):
        """准备训练数据（合并原有数据和用户上传数据）"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 创建输出子目录
        for split in ["train", "val", "test"]:
            (output_dir / split).mkdir(exist_ok=True)
            for category in self.categories.keys():
                (output_dir / split / category).mkdir(exist_ok=True)
        
        # 复制原有训练数据
        original_dir = Path("training_dataset")
        if original_dir.exists():
            for split in ["train", "val", "test"]:
                split_dir = original_dir / split
                if split_dir.exists():
                    for category in self.categories.keys():
                        category_dir = split_dir / category
                        if category_dir.exists():
                            target_dir = output_dir / split / category
                            for img_file in category_dir.glob("*"):
                                shutil.copy2(img_file, target_dir / img_file.name)
        
        # 添加用户上传的数据到训练集
        for category in self.categories.keys():
            category_dir = self.upload_dir / category
            if category_dir.exists():
                target_dir = output_dir / "train" / category
                for img_file in category_dir.glob("*"):
                    if img_file.is_file():
                        shutil.copy2(img_file, target_dir / f"user_{img_file.name}")
        
        logger.info(f"训练数据已准备完成: {output_dir}")
        return output_dir

def render_upload_interface():
    """渲染上传界面"""
    st.markdown("## 📸 图片上传和标注系统")
    st.markdown("请上传壁画图片并标注病害类型，帮助我们扩充训练数据集！")
    
    # 创建上传系统
    upload_system = ImageUploadSystem()
    
    # 显示当前统计
    st.markdown("### 📊 当前数据集统计")
    stats = upload_system.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总图片数", stats['total'])
    with col2:
        st.metric("裂缝病害", stats['crack'])
    with col3:
        st.metric("剥落病害", stats['peel'])
    with col4:
        st.metric("完好壁画", stats['clean'])
    
    # 上传界面
    st.markdown("### 📤 上传新图片")
    
    # 选择类别
    category_options = {v: k for k, v in upload_system.categories.items()}
    selected_category_display = st.selectbox(
        "选择病害类型",
        options=list(category_options.keys()),
        help="请根据图片内容选择最合适的病害类型"
    )
    selected_category = category_options[selected_category_display]
    
    # 上传文件
    uploaded_file = st.file_uploader(
        "选择图片文件",
        type=['jpg', 'jpeg', 'png'],
        help="支持 JPG、JPEG、PNG 格式，建议图片清晰，病害特征明显"
    )
    
    # 描述信息
    description = st.text_area(
        "图片描述（可选）",
        placeholder="请描述图片中的病害特征、严重程度等信息...",
        help="详细的描述有助于提高模型训练效果"
    )
    
    # 预览和上传
    if uploaded_file is not None:
        # 显示预览
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**图片预览**")
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_column_width=True)
        
        with col2:
            st.markdown("**上传信息**")
            st.write(f"**文件名**: {uploaded_file.name}")
            st.write(f"**文件大小**: {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**图片尺寸**: {image.size}")
            st.write(f"**选择类别**: {selected_category_display}")
            if description:
                st.write(f"**描述**: {description}")
        
        # 上传按钮
        if st.button("📤 上传图片", type="primary"):
            with st.spinner("正在上传图片..."):
                success, result = upload_system.upload_image(
                    uploaded_file, selected_category, description
                )
            
            if success:
                st.success(f"✅ 图片上传成功！文件名: {result}")
                st.rerun()
            else:
                st.error(f"❌ 上传失败: {result}")
    
    # 显示已上传的图片
    st.markdown("### 📋 已上传图片列表")
    df = upload_system.get_annotation_dataframe()
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        # 数据导出
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 导出标注数据"):
                csv_data = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="下载CSV文件",
                    data=csv_data,
                    file_name=f"mural_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🔄 准备训练数据"):
                with st.spinner("正在准备训练数据..."):
                    output_dir = upload_system.prepare_training_data()
                st.success(f"✅ 训练数据已准备完成！输出目录: {output_dir}")
                st.info("💡 现在可以运行 `python retrain_expanded_model.py` 来重新训练模型")
    else:
        st.info("还没有上传任何图片，请先上传一些壁画图片来扩充数据集。")

def main():
    """主函数"""
    st.set_page_config(
        page_title="壁画图片上传系统",
        page_icon="📸",
        layout="wide"
    )
    
    st.title("📸 壁画病害图片上传和标注系统")
    st.markdown("---")
    
    render_upload_interface()

if __name__ == "__main__":
    main()










