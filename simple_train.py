#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的壁画病害分类训练脚本
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import json
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_simple_features(image):
    """提取简单特征"""
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

def load_data(data_dir):
    """加载数据"""
    print("加载数据...")
    
    data_dir = Path(data_dir)
    class_names = ["crack", "peel", "disc", "clean"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    X = []
    y = []
    
    for split in ["train", "val"]:
        split_dir = data_dir / split
        print(f"处理 {split} 数据...")
        
        for class_name in class_names:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        features = extract_simple_features(image)
                        X.append(features)
                        y.append(class_to_idx[class_name])
                        print(f"  加载: {img_path.name} -> {class_name}")
                    except Exception as e:
                        print(f"  跳过: {img_path.name} - {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"总共加载了 {len(X)} 个样本")
    print(f"特征维度: {X.shape[1]}")
    print(f"类别分布: {np.bincount(y)}")
    
    return X, y

def train_model(X, y):
    """训练模型"""
    print("训练模型...")
    
    # 创建随机森林分类器
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # 训练
    model.fit(X, y)
    
    # 预测
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"训练准确率: {accuracy:.4f}")
    
    # 分类报告
    class_names = ["crack", "peel", "disc", "clean"]
    report = classification_report(y, y_pred, target_names=class_names)
    print(f"\n分类报告:\n{report}")
    
    return model

def test_model(model, data_dir):
    """测试模型"""
    print("测试模型...")
    
    data_dir = Path(data_dir)
    class_names = ["crack", "peel", "disc", "clean"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    X_test = []
    y_test = []
    
    test_dir = data_dir / "test"
    for class_name in class_names:
        class_dir = test_dir / class_name
        if class_dir.exists():
            for img_path in class_dir.glob("*.jpg"):
                try:
                    image = Image.open(img_path).convert('RGB')
                    features = extract_simple_features(image)
                    X_test.append(features)
                    y_test.append(class_to_idx[class_name])
                    print(f"  测试: {img_path.name} -> {class_name}")
                except Exception as e:
                    print(f"  跳过: {img_path.name} - {e}")
    
    if X_test:
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"测试准确率: {accuracy:.4f}")
        
        # 分类报告
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(f"\n测试分类报告:\n{report}")
    else:
        print("没有测试数据")

def save_model(model, model_dir="simple_models"):
    """保存模型"""
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    # 保存模型
    model_path = model_dir / "mural_classifier.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"模型已保存到: {model_path}")

def main():
    print("简化壁画病害分类训练")
    print("="*50)
    
    # 检查数据目录
    data_dir = Path("training_dataset")
    if not data_dir.exists():
        print(f"错误: 数据目录不存在 {data_dir}")
        return
    
    try:
        # 加载数据
        X, y = load_data(data_dir)
        
        if len(X) == 0:
            print("没有加载到任何数据！")
            return
        
        # 训练模型
        model = train_model(X, y)
        
        # 测试模型
        test_model(model, data_dir)
        
        # 保存模型
        save_model(model)
        
        print("\n训练完成！")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



















