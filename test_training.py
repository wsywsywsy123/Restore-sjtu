#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练脚本
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

def test_data_loading():
    """测试数据加载"""
    print("测试数据加载...")
    
    data_dir = Path("training_dataset")
    print(f"数据目录: {data_dir}")
    print(f"数据目录存在: {data_dir.exists()}")
    
    if not data_dir.exists():
        print("数据目录不存在！")
        return False
    
    # 检查各个分割目录
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        print(f"\n{split} 目录: {split_dir}")
        print(f"存在: {split_dir.exists()}")
        
        if split_dir.exists():
            for category in ["crack", "peel", "disc", "clean"]:
                category_dir = split_dir / category
                if category_dir.exists():
                    images = list(category_dir.glob("*.jpg"))
                    print(f"  {category}: {len(images)} 张图片")
                    
                    # 测试加载第一张图片
                    if images:
                        try:
                            img = Image.open(images[0]).convert('RGB')
                            print(f"    第一张图片尺寸: {img.size}")
                        except Exception as e:
                            print(f"    加载图片失败: {e}")
                else:
                    print(f"  {category}: 目录不存在")
    
    return True

def test_feature_extraction():
    """测试特征提取"""
    print("\n测试特征提取...")
    
    try:
        from train_sklearn_classifier import MuralFeatureExtractor
        
        extractor = MuralFeatureExtractor()
        print("特征提取器创建成功")
        
        # 创建一个测试图片
        test_image = Image.new('RGB', (224, 224), (255, 0, 0))
        
        # 提取特征
        features = extractor.extract_features(test_image)
        print(f"特征维度: {len(features)}")
        print(f"前10个特征: {features[:10]}")
        
        return True
        
    except Exception as e:
        print(f"特征提取测试失败: {e}")
        return False

def test_sklearn_imports():
    """测试sklearn导入"""
    print("\n测试sklearn导入...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.model_selection import cross_val_score
        print("sklearn导入成功")
        return True
    except Exception as e:
        print(f"sklearn导入失败: {e}")
        return False

def main():
    print("壁画病害分类训练测试")
    print("="*50)
    
    # 测试1: 数据加载
    if not test_data_loading():
        return
    
    # 测试2: sklearn导入
    if not test_sklearn_imports():
        return
    
    # 测试3: 特征提取
    if not test_feature_extraction():
        return
    
    print("\n所有测试通过！")
    print("可以开始训练了...")

if __name__ == "__main__":
    main()
