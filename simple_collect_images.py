#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版壁画病害图片收集脚本
使用requests而不是aiohttp，更稳定
"""

import os
import requests
import hashlib
from pathlib import Path
from PIL import Image
import pandas as pd
from duckduckgo_search import DDGS
import time
import random
from typing import List, Dict
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleMuralImageCollector:
    def __init__(self, output_dir: str = "mural_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 病害类型和对应的搜索关键词
        self.disease_categories = {
            "crack": {
                "keywords": [
                    "壁画裂缝", "壁画裂纹", "mural crack", "fresco crack",
                    "壁画龟裂", "壁画开裂", "cracked mural", "fresco fissure"
                ],
                "description": "裂缝病害"
            },
            "peel": {
                "keywords": [
                    "壁画剥落", "壁画脱落", "mural peeling", "fresco peeling",
                    "壁画起皮", "壁画剥蚀", "mural flaking", "fresco flaking"
                ],
                "description": "剥落病害"
            },
            "discoloration": {
                "keywords": [
                    "壁画变色", "壁画褪色", "mural discoloration", "fresco discoloration",
                    "壁画色彩变化", "壁画颜色褪变", "mural color change", "fresco fading"
                ],
                "description": "变色病害"
            },
            "stain_mold": {
                "keywords": [
                    "壁画污渍", "壁画霉斑", "mural stain", "fresco mold",
                    "壁画污染", "壁画霉变", "mural contamination", "fresco mildew"
                ],
                "description": "污渍霉斑"
            },
            "salt_weathering": {
                "keywords": [
                    "壁画盐蚀", "壁画风化", "mural salt damage", "fresco weathering",
                    "壁画盐析", "壁画风化", "mural efflorescence", "fresco erosion"
                ],
                "description": "盐蚀风化"
            },
            "bio_growth": {
                "keywords": [
                    "壁画生物附着", "壁画苔藓", "mural biological growth", "fresco moss",
                    "壁画微生物", "壁画藻类", "mural algae", "fresco lichen"
                ],
                "description": "生物附着"
            },
            "disc": {
                "keywords": [
                    "壁画脱落", "壁画缺失", "mural loss", "fresco loss",
                    "壁画缺损", "壁画破坏", "mural damage", "fresco damage"
                ],
                "description": "脱落缺损"
            },
            "clean": {
                "keywords": [
                    "完好壁画", "健康壁画", "intact mural", "healthy fresco",
                    "保存完好壁画", "无病害壁画", "well preserved mural", "undamaged fresco"
                ],
                "description": "完好壁画"
            }
        }
        
        # 创建分类目录
        for category in self.disease_categories.keys():
            (self.output_dir / category).mkdir(exist_ok=True)
        
        # 图片质量要求
        self.min_size = (224, 224)
        self.max_size = (2048, 2048)
        self.min_file_size = 10 * 1024  # 10KB
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        
        # 已下载图片的哈希值集合
        self.downloaded_hashes = set()
        self.load_existing_hashes()
        
        # 统计信息
        self.stats = {category: 0 for category in self.disease_categories.keys()}
        
    def load_existing_hashes(self):
        """加载已下载图片的哈希值"""
        hash_file = self.output_dir / "downloaded_hashes.txt"
        if hash_file.exists():
            with open(hash_file, 'r', encoding='utf-8') as f:
                self.downloaded_hashes = set(line.strip() for line in f if line.strip())
            logger.info(f"加载了 {len(self.downloaded_hashes)} 个已下载图片的哈希值")
    
    def save_hash(self, image_hash: str):
        """保存图片哈希值"""
        hash_file = self.output_dir / "downloaded_hashes.txt"
        with open(hash_file, 'a', encoding='utf-8') as f:
            f.write(f"{image_hash}\n")
        self.downloaded_hashes.add(image_hash)
    
    def calculate_image_hash(self, image_path: str) -> str:
        """计算图片的感知哈希值"""
        try:
            with Image.open(image_path) as img:
                img_gray = img.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
                pixels = list(img_gray.getdata())
                avg = sum(pixels) / len(pixels)
                hash_bits = ''.join('1' if pixel > avg else '0' for pixel in pixels)
                return hashlib.md5(hash_bits.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"计算图片哈希失败 {image_path}: {e}")
            return ""
    
    def is_image_duplicate(self, image_path: str) -> bool:
        """检查图片是否重复"""
        image_hash = self.calculate_image_hash(image_path)
        if not image_hash or image_hash in self.downloaded_hashes:
            return True
        return False
    
    def validate_image(self, image_path: str) -> bool:
        """验证图片质量"""
        try:
            with Image.open(image_path) as img:
                # 检查尺寸
                if img.size[0] < self.min_size[0] or img.size[1] < self.min_size[1]:
                    return False
                if img.size[0] > self.max_size[0] or img.size[1] > self.max_size[1]:
                    return False
                
                # 检查文件大小
                file_size = os.path.getsize(image_path)
                if file_size < self.min_file_size or file_size > self.max_file_size:
                    return False
                
                # 检查是否为有效图片
                img.verify()
                return True
        except Exception as e:
            logger.warning(f"图片验证失败 {image_path}: {e}")
            return False
    
    def download_image(self, url: str, category: str, filename: str) -> bool:
        """下载单张图片"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 200:
                image_path = self.output_dir / category / filename
                
                # 保存图片
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                # 验证图片
                if self.validate_image(str(image_path)):
                    # 检查是否重复
                    if not self.is_image_duplicate(str(image_path)):
                        # 保存哈希值
                        image_hash = self.calculate_image_hash(str(image_path))
                        if image_hash:
                            self.save_hash(image_hash)
                            self.stats[category] += 1
                            logger.info(f"成功下载 {category}: {filename}")
                            return True
                    else:
                        logger.info(f"跳过重复图片: {filename}")
                else:
                    logger.warning(f"图片质量不符合要求: {filename}")
                
                # 删除不符合要求的图片
                if image_path.exists():
                    image_path.unlink()
                        
        except Exception as e:
            logger.warning(f"下载图片失败 {url}: {e}")
        
        return False
    
    def search_and_download_category(self, category: str, keywords: List[str], 
                                   max_images: int = 20) -> int:
        """搜索并下载指定类别的图片"""
        logger.info(f"开始收集 {category} 类图片...")
        
        downloaded_count = 0
        search_terms = random.sample(keywords, min(2, len(keywords)))  # 随机选择2个关键词
        
        for search_term in search_terms:
            if downloaded_count >= max_images:
                break
            
            logger.info(f"搜索关键词: {search_term}")
            
            try:
                # 使用DuckDuckGo搜索图片
                with DDGS() as ddgs:
                    results = list(ddgs.images(
                        search_term,
                        max_results=15,
                        safesearch="moderate"
                    ))
                
                # 下载图片
                for i, result in enumerate(results):
                    if downloaded_count >= max_images:
                        break
                    
                    url = result.get('image')
                    if url:
                        filename = f"{category}_{int(time.time())}_{i}.jpg"
                        if self.download_image(url, category, filename):
                            downloaded_count += 1
                        
                        # 添加延迟避免被限制
                        time.sleep(1)
                        
            except Exception as e:
                logger.error(f"搜索失败 {search_term}: {e}")
            
            # 搜索间添加延迟
            time.sleep(2)
        
        logger.info(f"{category} 类图片收集完成，下载了 {downloaded_count} 张")
        return downloaded_count
    
    def collect_all_categories(self, max_images_per_category: int = 15):
        """收集所有类别的图片"""
        logger.info("开始收集壁画病害图片...")
        
        total_downloaded = 0
        for category, info in self.disease_categories.items():
            downloaded = self.search_and_download_category(
                category, 
                info["keywords"], 
                max_images_per_category
            )
            total_downloaded += downloaded
            
            # 类别间添加延迟
            time.sleep(3)
        
        logger.info(f"图片收集完成！总共下载了 {total_downloaded} 张图片")
        return total_downloaded
    
    def create_dataset_manifest(self):
        """创建数据集清单"""
        manifest_data = []
        
        for category in self.disease_categories.keys():
            category_dir = self.output_dir / category
            if category_dir.exists():
                for image_file in category_dir.glob("*.jpg"):
                    manifest_data.append({
                        "filename": image_file.name,
                        "category": category,
                        "description": self.disease_categories[category]["description"],
                        "path": str(image_file.relative_to(self.output_dir)),
                        "size": os.path.getsize(image_file)
                    })
        
        # 保存为CSV
        df = pd.DataFrame(manifest_data)
        manifest_file = self.output_dir / "dataset_manifest.csv"
        df.to_csv(manifest_file, index=False, encoding='utf-8-sig')
        
        # 保存统计信息
        stats_file = self.output_dir / "collection_stats.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("壁画病害图片收集统计\n")
            f.write("=" * 50 + "\n")
            f.write(f"总图片数: {len(manifest_data)}\n")
            f.write(f"收集时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for category, count in self.stats.items():
                f.write(f"{self.disease_categories[category]['description']}: {count} 张\n")
        
        logger.info(f"数据集清单已保存到: {manifest_file}")
        logger.info(f"统计信息已保存到: {stats_file}")
        
        return manifest_file

def main():
    """主函数"""
    print("壁画病害图片收集器 (简化版)")
    print("=" * 50)
    
    # 创建收集器
    collector = SimpleMuralImageCollector()
    
    # 显示配置信息
    print(f"输出目录: {collector.output_dir}")
    print(f"病害类别: {len(collector.disease_categories)} 种")
    for category, info in collector.disease_categories.items():
        print(f"  - {info['description']}: {len(info['keywords'])} 个关键词")
    
    print(f"\n图片质量要求:")
    print(f"  - 最小尺寸: {collector.min_size}")
    print(f"  - 最大尺寸: {collector.max_size}")
    print(f"  - 文件大小: {collector.min_file_size//1024}KB - {collector.max_file_size//1024//1024}MB")
    
    # 开始收集
    try:
        total_downloaded = collector.collect_all_categories(max_images_per_category=10)
        
        # 创建数据集清单
        manifest_file = collector.create_dataset_manifest()
        
        print(f"\n收集完成！")
        print(f"总共下载: {total_downloaded} 张图片")
        print(f"数据集清单: {manifest_file}")
        
    except KeyboardInterrupt:
        print("\n用户中断收集过程")
    except Exception as e:
        print(f"\n收集过程中出现错误: {e}")
        logger.exception("收集过程异常")

if __name__ == "__main__":
    main()

