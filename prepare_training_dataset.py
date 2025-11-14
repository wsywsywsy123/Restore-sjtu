#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备壁画病害训练数据集
整理现有图片，创建训练、验证、测试集
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image
import pandas as pd
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingDatasetPreparer:
    def __init__(self, source_dir="dataset_raw", output_dir="training_dataset"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建输出子目录
        for split in ["train", "val", "test"]:
            (self.output_dir / split).mkdir(exist_ok=True)
            for category in ["crack", "peel", "disc", "clean"]:
                (self.output_dir / split / category).mkdir(exist_ok=True)
        
        # 病害类别映射
        self.category_mapping = {
            "crack": "crack",
            "peel": "peel", 
            "disc": "disc",
            "clean": "clean"
        }
        
        # 数据集分割比例
        self.split_ratios = {
            "train": 0.7,
            "val": 0.2,
            "test": 0.1
        }
        
        # 统计信息
        self.stats = {
            "total_images": 0,
            "train": 0,
            "val": 0,
            "test": 0,
            "categories": {}
        }
    
    def validate_image(self, image_path: Path) -> bool:
        """验证图片是否有效"""
        try:
            with Image.open(image_path) as img:
                # 检查尺寸
                if img.size[0] < 64 or img.size[1] < 64:
                    return False
                # 检查是否为有效图片
                img.verify()
                return True
        except Exception as e:
            logger.warning(f"图片验证失败 {image_path}: {e}")
            return False
    
    def resize_image(self, image_path: Path, target_size=(224, 224)) -> bool:
        """调整图片尺寸"""
        try:
            with Image.open(image_path) as img:
                # 保持宽高比
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # 创建正方形画布
                new_img = Image.new('RGB', target_size, (255, 255, 255))
                
                # 居中粘贴
                x = (target_size[0] - img.size[0]) // 2
                y = (target_size[1] - img.size[1]) // 2
                new_img.paste(img, (x, y))
                
                # 保存
                new_img.save(image_path)
                return True
        except Exception as e:
            logger.warning(f"图片调整失败 {image_path}: {e}")
            return False
    
    def prepare_category_data(self, category: str) -> list:
        """准备指定类别的数据"""
        category_dir = self.source_dir / category
        if not category_dir.exists():
            logger.warning(f"类别目录不存在: {category_dir}")
            return []
        
        valid_images = []
        
        # 遍历所有图片文件
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            for image_path in category_dir.glob(ext):
                if self.validate_image(image_path):
                    valid_images.append(image_path)
                else:
                    logger.info(f"跳过无效图片: {image_path}")
        
        logger.info(f"类别 {category}: 找到 {len(valid_images)} 张有效图片")
        return valid_images
    
    def split_dataset(self, images: list) -> dict:
        """分割数据集"""
        random.shuffle(images)
        
        total = len(images)
        train_end = int(total * self.split_ratios["train"])
        val_end = train_end + int(total * self.split_ratios["val"])
        
        return {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }
    
    def copy_images(self, images: list, split: str, category: str):
        """复制图片到指定分割和类别目录"""
        target_dir = self.output_dir / split / category
        target_dir.mkdir(exist_ok=True)
        
        copied_count = 0
        for i, image_path in enumerate(images):
            try:
                # 生成新文件名
                new_filename = f"{category}_{split}_{i:04d}{image_path.suffix}"
                target_path = target_dir / new_filename
                
                # 复制文件
                shutil.copy2(image_path, target_path)
                
                # 调整尺寸
                if self.resize_image(target_path):
                    copied_count += 1
                    self.stats[split] += 1
                    self.stats["total_images"] += 1
                else:
                    # 如果调整失败，删除文件
                    target_path.unlink()
                    
            except Exception as e:
                logger.warning(f"复制图片失败 {image_path}: {e}")
        
        logger.info(f"成功复制 {copied_count} 张图片到 {split}/{category}")
        return copied_count
    
    def prepare_all_data(self):
        """准备所有数据"""
        logger.info("开始准备训练数据集...")
        
        all_data = {}
        
        # 处理每个类别
        for category in self.category_mapping.keys():
            logger.info(f"处理类别: {category}")
            
            # 准备数据
            images = self.prepare_category_data(category)
            if not images:
                logger.warning(f"类别 {category} 没有有效图片")
                continue
            
            # 分割数据集
            split_data = self.split_dataset(images)
            all_data[category] = split_data
            
            # 复制图片
            for split in ["train", "val", "test"]:
                if split_data[split]:
                    copied = self.copy_images(split_data[split], split, category)
                    if category not in self.stats["categories"]:
                        self.stats["categories"][category] = {}
                    self.stats["categories"][category][split] = copied
        
        return all_data
    
    def create_dataset_manifest(self, all_data: dict):
        """创建数据集清单"""
        manifest_data = []
        
        for category, split_data in all_data.items():
            for split, images in split_data.items():
                for i, image_path in enumerate(images):
                    manifest_data.append({
                        "original_path": str(image_path),
                        "category": category,
                        "split": split,
                        "filename": f"{category}_{split}_{i:04d}{image_path.suffix}",
                        "new_path": f"{split}/{category}/{category}_{split}_{i:04d}{image_path.suffix}"
                    })
        
        # 保存为CSV
        df = pd.DataFrame(manifest_data)
        manifest_file = self.output_dir / "dataset_manifest.csv"
        df.to_csv(manifest_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"数据集清单已保存到: {manifest_file}")
        return manifest_file
    
    def create_training_config(self):
        """创建训练配置文件"""
        config = {
            "dataset_name": "壁画病害分类数据集",
            "num_classes": len(self.category_mapping),
            "class_names": list(self.category_mapping.keys()),
            "image_size": [224, 224],
            "num_channels": 3,
            "splits": {
                "train": self.stats["train"],
                "val": self.stats["val"],
                "test": self.stats["test"]
            },
            "categories": self.stats["categories"]
        }
        
        import json
        config_file = self.output_dir / "dataset_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练配置已保存到: {config_file}")
        return config_file
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print("数据集准备完成！")
        print("="*50)
        print(f"总图片数: {self.stats['total_images']}")
        print(f"训练集: {self.stats['train']} 张")
        print(f"验证集: {self.stats['val']} 张")
        print(f"测试集: {self.stats['test']} 张")
        print("\n各类别统计:")
        
        for category, splits in self.stats["categories"].items():
            print(f"\n{category}:")
            for split, count in splits.items():
                print(f"  {split}: {count} 张")
        
        print(f"\n输出目录: {self.output_dir}")
        print("="*50)

def main():
    """主函数"""
    print("壁画病害训练数据集准备器")
    print("="*50)
    
    # 检查源目录
    source_dir = Path("dataset_raw")
    if not source_dir.exists():
        print(f"错误: 源目录不存在 {source_dir}")
        return
    
    # 创建准备器
    preparer = TrainingDatasetPreparer()
    
    # 准备数据
    try:
        all_data = preparer.prepare_all_data()
        
        # 创建清单和配置
        preparer.create_dataset_manifest(all_data)
        preparer.create_training_config()
        
        # 打印统计信息
        preparer.print_statistics()
        
    except Exception as e:
        logger.error(f"准备数据集时出现错误: {e}")
        raise

if __name__ == "__main__":
    main()



















