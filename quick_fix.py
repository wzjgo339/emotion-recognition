#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速修复模型性能问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
import numpy as np
from collections import Counter

from model import CNNWithAttention
from data_loader import get_data_loaders, FER2013Dataset
from torchvision import transforms

def create_balanced_sampler(dataset):
    """创建平衡采样器解决数据不平衡"""
    # 统计各类别数量
    targets = [label for _, label in dataset]
    class_counts = Counter(targets)
    
    # 计算各类别的权重
    class_weights = {cls: len(targets) / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in targets]
    
    # 创建采样器
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def get_balanced_data_loaders(data_dir='.', batch_size=32):
    """创建平衡的数据加载器"""
    
    # 改进的数据增强 - 对少数类加强增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # 增加旋转角度
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),  # 增加平移
        transforms.RandomResizedCrop(48, scale=(0.85, 1.0)),  # 增加尺度变化
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色扰动
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),  # 随机模糊
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5076], std=[0.2128])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5076], std=[0.2128])
    ])
    
    # 创建数据集
    train_dataset = FER2013Dataset(data_dir, transform=train_transform, mode='train')
    val_dataset = FER2013Dataset(data_dir, transform=val_test_transform, mode='val')
    test_dataset = FER2013Dataset(data_dir, transform=val_test_transform, mode='test')
    
    # 创建平衡采样器
    train_sampler = create_balanced_sampler(train_dataset)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,  # 使用平衡采样器
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

class WeightedCrossEntropyLoss(nn.Module):
    """加权交叉熵损失"""
    def __init__(self, dataset, device):
        super(WeightedCrossEntropyLoss, self).__init__()
        
        # 计算类别权重
        targets = [label for _, label in dataset]
        class_counts = Counter(targets)
        total_samples = len(targets)
        
        # 使用反比权重
        weights = []
        for i in range(7):  # 7个类别
            count = class_counts.get(i, 0)
            weight = total_samples / (7 * count) if count > 0 else 0
            weights.append(weight)
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.sum() * 7
        
        self.weights = torch.FloatTensor(weights).to(device)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights)
        
        print("类别权重:", dict(zip(range(7), weights)))
    
    def forward(self, inputs, targets):
        return self.ce_loss(inputs, targets)

def quick_train():
    """快速训练改进模型"""
    print("开始快速修复训练...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取平衡的数据加载器
    train_loader, val_loader, test_loader = get_balanced_data_loaders(batch_size=64)
    
    # 创建模型
    model = CNNWithAttention(num_classes=7).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 加载预训练权重（如果存在）
    if os.path.exists('best_model.pth'):
        try:
            checkpoint = torch.load('best_model.pth', map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("成功加载预训练权重")
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    
    # 创建加权损失函数
    train_dataset = FER2013Dataset('.', transform=None, mode='train')
    criterion = WeightedCrossEntropyLoss(train_dataset, device)
    
    # 优化器 - 使用更小的学习率进行微调
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # 训练参数
    num_epochs = 20  # 快速训练
    best_val_acc = 0.0
    
    print(f"开始训练 {num_epochs} 轮...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        train_acc = 100. * correct / total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f'学习率调整: {old_lr:.6f} -> {new_lr:.6f}')
        
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {new_lr:.6f}')
        print('-' * 60)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'best_fixed_model.pth')
            print(f'新的最佳模型已保存! 验证准确率: {val_acc:.2f}%')
        
        # 早停机制
        if epoch >= 5 and val_acc < best_val_acc - 5:
            print("验证准确率持续下降，提前停止训练")
            break
    
    print(f"快速修复训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    print("最佳模型已保存为: best_fixed_model.pth")
    
    # 测试模型
    test_model_performance(best_val_acc)

def test_model_performance(best_val_acc):
    """测试修复后的模型"""
    print("\n开始测试修复后的模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载最佳模型
    model = CNNWithAttention(num_classes=7).to(device)
    checkpoint = torch.load('best_fixed_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 获取测试数据
    _, _, test_loader = get_balanced_data_loaders(batch_size=64)
    
    # 测试
    test_correct = 0
    test_total = 0
    class_correct = [0] * 7
    class_total = [0] * 7
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # 统计各类别准确率
            for i in range(labels.size(0)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    test_acc = 100. * test_correct / test_total
    
    print(f"\n测试结果:")
    print(f"总体测试准确率: {test_acc:.2f}%")
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    print(f"\n各类别测试准确率:")
    for i, emotion in enumerate(emotions):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"  {emotion:8s}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  {emotion:8s}: 0.00% (0/0)")
    
    print(f"\n验证准确率: {best_val_acc:.2f}%")
    print(f"测试准确率: {test_acc:.2f}%")
    
    if test_acc > 50:
        print("✅ 模型修复成功！准确率有明显提升")
    else:
        print("⚠️ 模型仍需进一步优化，建议使用完整的改进训练脚本")

if __name__ == "__main__":
    import os
    quick_train()