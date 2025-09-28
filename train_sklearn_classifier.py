#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
壁画病害分类模型训练脚本 (scikit-learn版本)
使用传统机器学习方法进行分类
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
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MuralFeatureExtractor:
    """壁画特征提取器"""
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
    
    def extract_color_features(self, image):
        """提取颜色特征"""
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 计算RGB通道的统计特征
        features = []
        
        for channel in range(3):  # R, G, B
            channel_data = img_array[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75)
            ])
        
        # 计算HSV特征
        hsv_image = image.convert('HSV')
        hsv_array = np.array(hsv_image)
        
        for channel in range(3):  # H, S, V
            channel_data = hsv_array[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data)
            ])
        
        return features
    
    def extract_texture_features(self, image):
        """提取纹理特征"""
        # 转换为灰度图
        gray_image = image.convert('L')
        img_array = np.array(gray_image)
        
        features = []
        
        # 计算灰度统计特征
        features.extend([
            np.mean(img_array),
            np.std(img_array),
            np.var(img_array),
            np.median(img_array)
        ])
        
        # 计算梯度特征
        grad_x = np.gradient(img_array, axis=1)
        grad_y = np.gradient(img_array, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude)
        ])
        
        # 计算边缘密度
        edge_density = np.sum(gradient_magnitude > np.percentile(gradient_magnitude, 90)) / gradient_magnitude.size
        features.append(edge_density)
        
        return features
    
    def extract_histogram_features(self, image):
        """提取直方图特征"""
        # 转换为灰度图
        gray_image = image.convert('L')
        img_array = np.array(gray_image)
        
        # 计算直方图
        hist, _ = np.histogram(img_array, bins=32, range=(0, 256))
        hist = hist / hist.sum()  # 归一化
        
        # 提取直方图统计特征
        features = []
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
    
    def extract_features(self, image):
        """提取所有特征"""
        # 调整图片大小
        image = image.resize(self.image_size)
        
        features = []
        features.extend(self.extract_color_features(image))
        features.extend(self.extract_texture_features(image))
        features.extend(self.extract_histogram_features(image))
        
        return features

class MuralDataset:
    """壁画数据集"""
    
    def __init__(self, data_dir, feature_extractor):
        self.data_dir = Path(data_dir)
        self.feature_extractor = feature_extractor
        
        # 类别映射
        self.class_names = ["crack", "peel", "disc", "clean"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 数据
        self.X = []
        self.y = []
        self.samples = []
    
    def load_data(self, split):
        """加载指定分割的数据"""
        split_dir = self.data_dir / split
        
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        features = self.feature_extractor.extract_features(image)
                        
                        self.X.append(features)
                        self.y.append(self.class_to_idx[class_name])
                        self.samples.append((img_path, class_name))
                        
                    except Exception as e:
                        logger.warning(f"加载图片失败 {img_path}: {e}")
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        logger.info(f"加载了 {len(self.X)} 个 {split} 样本")
        logger.info(f"特征维度: {self.X.shape[1]}")
    
    def get_data(self):
        """获取数据"""
        return self.X, self.y

class MuralClassifierTrainer:
    """壁画分类器训练器"""
    
    def __init__(self, data_dir, model_dir="sklearn_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # 特征提取器
        self.feature_extractor = MuralFeatureExtractor()
        
        # 模型
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # 训练结果
        self.results = {}
    
    def train_models(self):
        """训练所有模型"""
        logger.info("开始训练模型...")
        
        # 加载训练数据
        train_dataset = MuralDataset(self.data_dir, self.feature_extractor)
        train_dataset.load_data("train")
        X_train, y_train = train_dataset.get_data()
        
        # 加载验证数据
        val_dataset = MuralDataset(self.data_dir, self.feature_extractor)
        val_dataset.load_data("val")
        X_val, y_val = val_dataset.get_data()
        
        # 训练每个模型
        for name, model in self.models.items():
            logger.info(f"训练 {name}...")
            
            # 训练
            model.fit(X_train, y_train)
            
            # 验证
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=3)
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            logger.info(f"{name} - 验证准确率: {accuracy:.4f}")
            logger.info(f"{name} - 交叉验证: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 选择最佳模型
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.results[best_model_name]['model']
        
        logger.info(f"最佳模型: {best_model_name}")
        
        # 保存模型
        self.save_models()
        
        return best_model, best_model_name
    
    def save_models(self):
        """保存模型"""
        for name, result in self.results.items():
            model_path = self.model_dir / f"{name.lower()}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
            logger.info(f"保存模型: {model_path}")
        
        # 保存特征提取器
        extractor_path = self.model_dir / "feature_extractor.pkl"
        with open(extractor_path, 'wb') as f:
            pickle.dump(self.feature_extractor, f)
        
        # 保存结果
        results_path = self.model_dir / "training_results.json"
        results_data = {}
        for name, result in self.results.items():
            results_data[name] = {
                'accuracy': result['accuracy'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
            }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练结果已保存: {results_path}")
    
    def test_model(self, model, model_name):
        """测试模型"""
        # 加载测试数据
        test_dataset = MuralDataset(self.data_dir, self.feature_extractor)
        test_dataset.load_data("test")
        X_test, y_test = test_dataset.get_data()
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        
        # 分类报告
        class_names = ["crack", "peel", "disc", "clean"]
        report = classification_report(y_test, y_pred, target_names=class_names)
        
        logger.info(f"\n{model_name} 测试结果:")
        logger.info(f"测试准确率: {accuracy:.4f}")
        logger.info(f"\n分类报告:\n{report}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} 混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig(self.model_dir / f"{model_name.lower()}_confusion_matrix.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"混淆矩阵已保存: {self.model_dir}/{model_name.lower()}_confusion_matrix.png")
        
        return accuracy, report

def main():
    """主函数"""
    print("壁画病害分类模型训练 (scikit-learn版本)")
    print("="*60)
    
    # 检查数据目录
    data_dir = Path("training_dataset")
    if not data_dir.exists():
        print(f"错误: 数据目录不存在 {data_dir}")
        print("请先运行 prepare_training_dataset.py 准备数据")
        return
    
    print(f"数据目录: {data_dir}")
    print(f"数据目录存在: {data_dir.exists()}")
    
    # 创建训练器
    trainer = MuralClassifierTrainer(data_dir)
    
    try:
        # 训练模型
        best_model, best_model_name = trainer.train_models()
        
        # 测试最佳模型
        print(f"\n测试最佳模型: {best_model_name}")
        accuracy, report = trainer.test_model(best_model, best_model_name)
        
        print(f"\n训练完成！")
        print(f"最佳模型: {best_model_name}")
        print(f"测试准确率: {accuracy:.4f}")
        print(f"模型保存在: {trainer.model_dir}")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
