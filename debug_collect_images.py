#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试版壁画病害图片收集脚本
"""

import os
import requests
import hashlib
from pathlib import Path
from PIL import Image
import pandas as pd
import time
import random
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_duckduckgo():
    """测试DuckDuckGo搜索"""
    try:
        from duckduckgo_search import DDGS
        print("DuckDuckGo搜索模块导入成功")
        
        with DDGS() as ddgs:
            results = list(ddgs.images("mural crack", max_results=5))
            print(f"搜索到 {len(results)} 个结果")
            for i, result in enumerate(results):
                print(f"结果 {i+1}: {result.get('image', 'No URL')}")
        return True
    except Exception as e:
        print(f"DuckDuckGo搜索失败: {e}")
        return False

def download_test_image():
    """测试下载图片"""
    try:
        # 使用一个公开的测试图片URL
        test_url = "https://via.placeholder.com/300x200/FF0000/FFFFFF?text=Test+Image"
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            test_dir = Path("test_download")
            test_dir.mkdir(exist_ok=True)
            
            test_file = test_dir / "test.jpg"
            with open(test_file, 'wb') as f:
                f.write(response.content)
            
            print(f"测试图片下载成功: {test_file}")
            
            # 验证图片
            with Image.open(test_file) as img:
                print(f"图片尺寸: {img.size}")
                print(f"图片模式: {img.mode}")
            
            return True
        else:
            print(f"下载失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"下载测试失败: {e}")
        return False

def simple_image_search():
    """简单的图片搜索和下载"""
    try:
        from duckduckgo_search import DDGS
        
        # 创建输出目录
        output_dir = Path("simple_dataset")
        output_dir.mkdir(exist_ok=True)
        
        # 简单的搜索词
        search_terms = ["mural crack", "fresco damage", "ancient painting"]
        
        downloaded_count = 0
        
        with DDGS() as ddgs:
            for term in search_terms:
                print(f"搜索: {term}")
                
                try:
                    results = list(ddgs.images(term, max_results=3))
                    print(f"找到 {len(results)} 个结果")
                    
                    for i, result in enumerate(results):
                        url = result.get('image')
                        if url:
                            print(f"尝试下载: {url}")
                            
                            try:
                                response = requests.get(url, timeout=15)
                                if response.status_code == 200:
                                    filename = f"{term.replace(' ', '_')}_{i}.jpg"
                                    filepath = output_dir / filename
                                    
                                    with open(filepath, 'wb') as f:
                                        f.write(response.content)
                                    
                                    # 验证图片
                                    try:
                                        with Image.open(filepath) as img:
                                            print(f"成功下载: {filename}, 尺寸: {img.size}")
                                            downloaded_count += 1
                                    except Exception as e:
                                        print(f"图片验证失败: {e}")
                                        filepath.unlink()
                                        
                            except Exception as e:
                                print(f"下载失败: {e}")
                                
                        time.sleep(1)  # 添加延迟
                        
                except Exception as e:
                    print(f"搜索失败 {term}: {e}")
                
                time.sleep(2)  # 搜索间延迟
        
        print(f"总共下载了 {downloaded_count} 张图片")
        return downloaded_count
        
    except Exception as e:
        print(f"简单搜索失败: {e}")
        return 0

def main():
    print("壁画图片收集调试脚本")
    print("=" * 50)
    
    # 测试1: DuckDuckGo搜索
    print("\n1. 测试DuckDuckGo搜索...")
    if test_duckduckgo():
        print("✓ DuckDuckGo搜索正常")
    else:
        print("✗ DuckDuckGo搜索失败")
        return
    
    # 测试2: 图片下载
    print("\n2. 测试图片下载...")
    if download_test_image():
        print("✓ 图片下载正常")
    else:
        print("✗ 图片下载失败")
        return
    
    # 测试3: 简单搜索和下载
    print("\n3. 开始简单搜索和下载...")
    count = simple_image_search()
    
    if count > 0:
        print(f"\n✓ 成功下载了 {count} 张图片")
    else:
        print("\n✗ 没有下载到任何图片")

if __name__ == "__main__":
    main()

