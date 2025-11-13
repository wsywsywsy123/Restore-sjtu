#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将真实照片转换为Base64编码的工具
支持案例库照片处理
"""
import base64
import os
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
from io import BytesIO

def image_to_base64(image_path: str, max_size: Optional[tuple] = None, quality: int = 85) -> str:
    """
    将图片文件转换为Base64编码
    
    参数:
        image_path: 图片文件路径
        max_size: 最大尺寸 (width, height)，None表示不缩放
        quality: JPEG质量 (1-100)
    
    返回:
        Base64编码的字符串（包含data URI前缀）
    """
    try:
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            return ""
        
        # 读取图片
        img = Image.open(image_path)
        
        # 如果是RGBA模式，转换为RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        
        # 缩放图片（如果需要）
        if max_size:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # 转换为字节
        buffer = BytesIO()
        img_format = img.format or 'JPEG'
        if img_format.upper() == 'PNG':
            img.save(buffer, format='PNG')
            mime_type = 'image/png'
        else:
            img.save(buffer, format='JPEG', quality=quality)
            mime_type = 'image/jpeg'
        
        buffer.seek(0)
        encoded_string = base64.b64encode(buffer.read()).decode('utf-8')
        
        # 返回data URI格式
        return f"data:{mime_type};base64,{encoded_string}"
    
    except Exception as e:
        print(f"图片转换错误: {e}")
        return ""

def base64_to_image(base64_string: str) -> Optional[Image.Image]:
    """
    将Base64编码转换为PIL Image对象
    
    参数:
        base64_string: Base64编码字符串（可以包含data URI前缀）
    
    返回:
        PIL Image对象
    """
    try:
        # 移除data URI前缀（如果存在）
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # 解码
        image_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(image_data))
        return img
    
    except Exception as e:
        print(f"Base64解码错误: {e}")
        return None

def process_case_photos(photo_dir: str = "case_photos", max_size: tuple = (1920, 1920)) -> Dict[str, str]:
    """
    批量处理案例照片目录中的所有图片
    
    参数:
        photo_dir: 照片目录路径
        max_size: 最大尺寸
    
    返回:
        字典，key为文件名（不含扩展名），value为Base64编码
    """
    photo_dir_path = Path(photo_dir)
    if not photo_dir_path.exists():
        print(f"照片目录不存在: {photo_dir}")
        return {}
    
    encoded_photos = {}
    supported_formats = ('.jpg', '.jpeg', '.png', '.webp')
    
    for image_file in photo_dir_path.iterdir():
        if image_file.suffix.lower() in supported_formats:
            key = image_file.stem  # 文件名（不含扩展名）
            base64_data = image_to_base64(str(image_file), max_size=max_size)
            if base64_data:
                encoded_photos[key] = base64_data
                print(f"已处理: {key}")
    
    return encoded_photos

def save_base64_to_file(base64_string: str, output_path: str):
    """
    将Base64编码保存为图片文件
    
    参数:
        base64_string: Base64编码字符串
        output_path: 输出文件路径
    """
    try:
        # 移除data URI前缀（如果存在）
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        print(f"图片已保存到: {output_path}")
    
    except Exception as e:
        print(f"保存图片错误: {e}")

def get_image_info(image_path: str) -> Dict:
    """
    获取图片信息
    
    返回:
        包含尺寸、格式、大小等信息的字典
    """
    try:
        img = Image.open(image_path)
        file_size = os.path.getsize(image_path)
        
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2)
        }
    except Exception as e:
        print(f"获取图片信息错误: {e}")
        return {}

if __name__ == "__main__":
    # 示例：处理单个图片
    print("照片处理工具示例")
    print("=" * 50)
    
    # 示例1: 转换单个图片
    # image_path = "test_image.jpg"
    # if os.path.exists(image_path):
    #     base64_data = image_to_base64(image_path, max_size=(1920, 1920))
    #     print(f"Base64长度: {len(base64_data)} 字符")
    
    # 示例2: 批量处理目录
    # photos = process_case_photos("case_photos")
    # print(f"共处理 {len(photos)} 张图片")
    
    print("\n使用方法:")
    print("1. 将照片放入 case_photos 目录")
    print("2. 运行: photos = process_case_photos('case_photos')")
    print("3. 使用返回的字典添加到案例库")


