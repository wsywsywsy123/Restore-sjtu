#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新训练扩充后的壁画病害分类模型
使用原有数据 + 用户上传数据
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import json
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_enhanced_features(image):
    """提取增强特征"""
    # 转换为numpy数组
    img_array = np.array(image)
    
    features = []
    
    # RGB通道统计特征
    for channel in range(3):
        channel_data = img_array[:, :, channel].flatten()
        features.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.median(channel_data),
            np.percentile(channel_data, 25),
            np.percentile(channel_data, 75)
        ])
    
    # HSV特征
    hsv_image = image.convert('HSV')
    hsv_array = np.array(hsv_image)
    
    for channel in range(3):  # H, S, V
        channel_data = hsv_array[:, :, channel].flatten()
        features.extend([
            np.mean(channel_data),
            np.std(channel_data)
        ])
    
    # 灰度统计特征
    gray = np.mean(img_array, axis=2)
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.var(gray),
        np.median(gray)
    ])
    
    # 梯度特征
    grad_x = np.gradient(gray, axis=1)
    grad_y = np.gradient(gray, axis=0)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    features.extend([
        np.mean(gradient_magnitude),
        np.std(gradient_magnitude)
    ])
    
    # 边缘密度
    edge_density = np.sum(gradient_magnitude > np.percentile(gradient_magnitude, 90)) / gradient_magnitude.size
    features.append(edge_density)
    
    # 直方图特征
    hist, _ = np.histogram(gray, bins=32, range=(0, 256))
    hist = hist / hist.sum()  # 归一化
    
    features.extend([
        np.mean(hist),
        np.std(hist),
        np.var(hist)
    ])
    
    # 计算直方图的峰值
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append(hist[i])
    
    if peaks:
        features.extend([
            np.mean(peaks),
            np.std(peaks),
            len(peaks)
        ])
    else:
        features.extend([0, 0, 0])
    
    return features

def load_expanded_data(data_dir):
    """加载扩充后的数据"""
    print("加载扩充后的数据...")
    
    data_dir = Path(data_dir)
    class_names = ["crack", "peel", "disc", "discoloration", "stain_mold", "salt_weathering", "bio_growth", "clean"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    X = []
    y = []
    file_paths = []
    
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
            
        print(f"处理 {split} 数据...")
        
        for class_name in class_names:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        features = extract_enhanced_features(image)
                        X.append(features)
                        y.append(class_to_idx[class_name])
                        file_paths.append(str(img_path))
                        print(f"  加载: {img_path.name} -> {class_name}")
                    except Exception as e:
                        print(f"  跳过: {img_path.name} - {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"总共加载了 {len(X)} 个样本")
    print(f"特征维度: {X.shape[1]}")
    print(f"类别分布: {np.bincount(y)}")
    
    return X, y, file_paths

def train_enhanced_models(X, y):
    """训练增强模型"""
    print("训练增强模型...")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 创建多个模型
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
        'SVM': SVC(kernel='rbf', C=1.0, random_state=42, probability=True),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest_Enhanced': RandomForestClassifier(
            n_estimators=300, 
            max_depth=20, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"训练 {name}...")
        
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"{name} - 测试准确率: {accuracy:.4f}")
        print(f"{name} - 交叉验证: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

def evaluate_models(results):
    """评估模型"""
    print("\n模型评估结果:")
    print("="*60)
    
    class_names = ["crack", "peel", "disc", "discoloration", "stain_mold", "salt_weathering", "bio_growth", "clean"]
    class_display_names = ["裂缝", "剥落", "脱落", "变色", "污渍", "盐蚀", "生物", "完好"]
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    print(f"最佳模型: {best_model_name}")
    print(f"最佳准确率: {results[best_model_name]['accuracy']:.4f}")
    
    # 详细评估最佳模型
    y_test = results[best_model_name]['y_test']
    y_pred = results[best_model_name]['y_pred']
    
    print(f"\n{best_model_name} 详细评估:")
    report = classification_report(y_test, y_pred, target_names=class_display_names)
    print(report)
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_display_names, yticklabels=class_display_names)
    plt.title(f'{best_model_name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('enhanced_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("混淆矩阵已保存: enhanced_confusion_matrix.png")
    
    return best_model, best_model_name

def save_enhanced_model(model, model_name, model_dir="enhanced_models"):
    """保存增强模型"""
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    # 保存模型
    model_path = model_dir / f"{model_name.lower()}_enhanced.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # 保存模型信息
    info = {
        "model_name": model_name,
        "feature_extractor": "extract_enhanced_features",
        "class_names": ["crack", "peel", "disc", "discoloration", "stain_mold", "salt_weathering", "bio_growth", "clean"],
        "class_display_names": ["裂缝", "剥落", "脱落", "变色", "污渍", "盐蚀", "生物", "完好"],
        "feature_dimension": 34,  # 根据extract_enhanced_features计算
        "training_date": str(datetime.now())
    }
    
    info_path = model_dir / f"{model_name.lower()}_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print(f"模型已保存: {model_path}")
    print(f"模型信息已保存: {info_path}")
    
    return model_path

def main():
    """主函数"""
    print("壁画病害分类模型重新训练（增强版）")
    print("="*60)
    
    # 检查扩充后的数据目录
    data_dir = Path("expanded_training_dataset")
    if not data_dir.exists():
        print(f"错误: 扩充数据目录不存在 {data_dir}")
        print("请先运行上传系统准备数据")
        return
    
    try:
        # 加载数据
        X, y, file_paths = load_expanded_data(data_dir)
        
        if len(X) == 0:
            print("没有加载到任何数据！")
            return
        
        # 训练模型
        results = train_enhanced_models(X, y)
        
        # 评估模型
        best_model, best_model_name = evaluate_models(results)
        
        # 保存最佳模型
        model_path = save_enhanced_model(best_model, best_model_name)
        
        print(f"\n训练完成！")
        print(f"最佳模型: {best_model_name}")
        print(f"模型路径: {model_path}")
        print(f"总样本数: {len(X)}")
        print(f"特征维度: {X.shape[1]}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from datetime import datetime
    main()













