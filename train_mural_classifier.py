#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
壁画病害分类模型训练脚本
使用预训练模型进行迁移学习
"""

import os
# 修复Windows PyTorch DLL加载问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from pathlib import Path
import logging
from PIL import Image
import numpy as np
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MuralDataset(Dataset):
    """壁画病害数据集"""
    
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # 类别映射
        self.class_names = ["crack", "peel", "disc", "clean"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 加载数据
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """加载样本"""
        split_dir = self.data_dir / self.split
        
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        logger.info(f"加载了 {len(self.samples)} 个 {self.split} 样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.warning(f"加载图片失败 {img_path}: {e}")
            # 返回一个空白图片
            blank_image = Image.new('RGB', (224, 224), (255, 255, 255))
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, 0

class MuralClassifier(nn.Module):
    """壁画病害分类器"""
    
    def __init__(self, num_classes=4, pretrained=True):
        super(MuralClassifier, self).__init__()
        
        # 使用预训练的ResNet18作为骨干网络
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 修改最后的分类层
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # 添加dropout防止过拟合
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return x

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, data_dir, model_dir="models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 数据变换
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def create_data_loaders(self, batch_size=8):
        """创建数据加载器"""
        # 训练集
        train_dataset = MuralDataset(self.data_dir, "train", self.train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # 验证集
        val_dataset = MuralDataset(self.data_dir, "val", self.val_transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # 测试集
        test_dataset = MuralDataset(self.data_dir, "test", self.val_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        logger.info(f"训练集: {len(train_dataset)} 样本")
        logger.info(f"验证集: {len(val_dataset)} 样本")
        logger.info(f"测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 5 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """验证一个epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs=20, batch_size=8, learning_rate=0.001):
        """训练模型"""
        logger.info("开始训练模型...")
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders(batch_size)
        
        # 创建模型
        model = MuralClassifier(num_classes=4, pretrained=True).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # 训练循环
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            logger.info(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.model_dir / "best_model.pth")
                logger.info(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        
        # 保存最终模型
        torch.save(model.state_dict(), self.model_dir / "final_model.pth")
        
        # 绘制训练历史
        self.plot_training_history()
        
        # 测试模型
        self.test_model(model, test_loader)
        
        logger.info("训练完成！")
        return model
    
    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='训练准确率')
        plt.plot(self.history['val_acc'], label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("训练历史图表已保存")
    
    def test_model(self, model, test_loader):
        """测试模型"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        class_names = ["crack", "peel", "disc", "clean"]
        
        # 分类报告
        report = classification_report(all_targets, all_preds, target_names=class_names)
        logger.info("测试集分类报告:")
        logger.info(f"\n{report}")
        
        # 混淆矩阵
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig(self.model_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("混淆矩阵已保存")

def main():
    """主函数"""
    print("壁画病害分类模型训练")
    print("="*50)
    
    # 检查数据目录
    data_dir = Path("training_dataset")
    if not data_dir.exists():
        print(f"错误: 数据目录不存在 {data_dir}")
        print("请先运行 prepare_training_dataset.py 准备数据")
        return
    
    # 创建训练器
    trainer = ModelTrainer(data_dir)
    
    # 开始训练
    try:
        model = trainer.train(epochs=15, batch_size=4, learning_rate=0.001)
        print("\n训练完成！")
        print(f"模型保存在: {trainer.model_dir}")
        print(f"训练历史图表: {trainer.model_dir}/training_history.png")
        print(f"混淆矩阵: {trainer.model_dir}/confusion_matrix.png")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()
