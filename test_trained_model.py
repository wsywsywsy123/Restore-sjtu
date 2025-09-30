#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练好的壁画病害分类模型
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import pickle

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

def load_model(model_path):
    """加载训练好的模型"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_image(model, image_path):
    """预测单张图片"""
    try:
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        
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
        print(f"预测失败 {image_path}: {e}")
        return None

def test_model_on_dataset(model, data_dir):
    """在数据集上测试模型"""
    print("在数据集上测试模型...")
    
    data_dir = Path(data_dir)
    class_names = ["crack", "peel", "disc", "clean"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    correct = 0
    total = 0
    
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
            
        print(f"\n{split} 集测试:")
        
        for class_name in class_names:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    result = predict_image(model, img_path)
                    if result:
                        predicted = result['predicted_class']
                        confidence = result['confidence']
                        
                        is_correct = predicted == class_name
                        if is_correct:
                            correct += 1
                        total += 1
                        
                        status = "✓" if is_correct else "✗"
                        print(f"  {status} {img_path.name}: {class_name} -> {predicted} ({confidence:.3f})")
    
    if total > 0:
        accuracy = correct / total
        print(f"\n总体准确率: {accuracy:.4f} ({correct}/{total})")
    else:
        print("没有找到测试图片")

def test_model_on_single_image(model, image_path):
    """测试单张图片"""
    print(f"测试图片: {image_path}")
    
    result = predict_image(model, image_path)
    if result:
        print(f"预测类别: {result['predicted_class']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("各类别概率:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
    else:
        print("预测失败")

def main():
    print("壁画病害分类模型测试")
    print("="*50)
    
    # 检查模型文件
    model_path = Path("simple_models/mural_classifier.pkl")
    if not model_path.exists():
        print(f"错误: 模型文件不存在 {model_path}")
        return
    
    # 加载模型
    print("加载模型...")
    model = load_model(model_path)
    print("模型加载成功")
    
    # 在数据集上测试
    data_dir = Path("training_dataset")
    if data_dir.exists():
        test_model_on_dataset(model, data_dir)
    else:
        print("数据集目录不存在，跳过数据集测试")
    
    # 测试单张图片（如果有的话）
    print("\n" + "="*50)
    print("单张图片测试:")
    
    # 查找一些测试图片
    test_images = []
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    for img_path in class_dir.glob("*.jpg"):
                        test_images.append(img_path)
                        if len(test_images) >= 3:  # 只测试前3张
                            break
                if len(test_images) >= 3:
                    break
        if len(test_images) >= 3:
            break
    
    for img_path in test_images[:3]:  # 测试前3张图片
        test_model_on_single_image(model, img_path)
        print()

if __name__ == "__main__":
    main()




