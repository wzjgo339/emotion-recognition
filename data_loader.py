import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


class FER2013Dataset(data.Dataset):
    """FER2013数据集加载器 - 支持文件夹格式"""
    def __init__(self, data_dir, transform=None, mode='train'):
        """
        Args:
            data_dir (string): 数据集根目录
            transform (callable, optional): 可选的图像变换
            mode (string): 'train', 'val', 或 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        # 表情类别映射
        self.emotion_map = {
            'angry': 0,
            'disgust': 1,
            'fear': 2,
            'happy': 3,
            'sad': 4,
            'surprise': 5,
            'neutral': 6
        }
        
        # 获取图像路径和标签
        self.images = []
        self.labels = []
        
        if mode == 'train':
            data_path = os.path.join(data_dir, 'train')
        elif mode == 'val':
            data_path = os.path.join(data_dir, 'test')  # 使用test作为验证集
        elif mode == 'test':
            data_path = os.path.join(data_dir, 'test')
        else:
            raise ValueError(f"未知的模式: {mode}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据路径不存在: {data_path}")
        
        # 遍历所有表情文件夹
        for emotion_name in os.listdir(data_path):
            emotion_path = os.path.join(data_path, emotion_name)
            if os.path.isdir(emotion_path) and emotion_name in self.emotion_map:
                # 获取该表情下的所有图像
                image_files = glob.glob(os.path.join(emotion_path, '*.jpg'))
                for img_path in image_files:
                    self.images.append(img_path)
                    self.labels.append(self.emotion_map[emotion_name])
        
        print(f"{mode}集样本数量: {len(self.images)}")
        
        # 打印各类别数量
        from collections import Counter
        label_counts = Counter(self.labels)
        emotion_distribution = {self.get_emotion_name(k): v for k, v in label_counts.items()}
        print(f"{mode}集各类别分布: {emotion_distribution}")
    
    def get_emotion_name(self, label):
        """根据标签获取表情名称"""
        emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        return emotion_names[label]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.images[idx]
        emotion = self.labels[idx]
        
        try:
            # 读取图像
            image = Image.open(img_path).convert('L')  # 转为灰度图
            
            # 确保图像尺寸为48x48
            if image.size != (48, 48):
                image = image.resize((48, 48), Image.LANCZOS)
            
            if self.transform:
                image = self.transform(image)
            
            return image, emotion
            
        except Exception as e:
            print(f"加载图像失败 {img_path}: {e}")
            # 返回一个默认图像
            default_image = torch.zeros(1, 48, 48)
            return default_image, emotion


def get_data_loaders(data_dir='.', batch_size=32, num_workers=4):
    """创建数据加载器 - 适配文件夹格式"""
    
    # 数据预处理和增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(48, scale=(0.9, 1.0)),
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
    
    # 对于验证集和测试集，我们可以分割test数据集
    # 或者使用相同的test数据作为验证和测试
    val_dataset = FER2013Dataset(data_dir, transform=val_test_transform, mode='val')
    test_dataset = FER2013Dataset(data_dir, transform=val_test_transform, mode='test')
    
    # 创建数据加载器
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def calculate_dataset_stats(data_dir='.'):
    """计算数据集的均值和标准差 - 适配文件夹格式"""
    dataset = FER2013Dataset(data_dir, transform=transforms.ToTensor(), mode='train')
    
    loader = data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)
    
    mean = 0.
    std = 0.
    total_images = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    
    mean /= total_images
    std /= total_images
    
    print(f"数据集均值: {mean}")
    print(f"数据集标准差: {std}")
    
    return mean, std


def visualize_data_distribution(data_dir='.'):
    """可视化训练集和测试集的表情分布饼图"""
    print("=" * 60)
    print("生成数据分布可视化图表")
    print("=" * 60)
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    
    # 加载训练集和测试集数据
    try:
        print("正在加载数据集...")
        
        # 创建数据集实例（不使用变换，只获取分布信息）
        train_dataset = FER2013Dataset(data_dir, transform=None, mode='train')
        test_dataset = FER2013Dataset(data_dir, transform=None, mode='test')
        
        # 统计训练集分布
        train_counts = Counter(train_dataset.labels)
        train_emotion_names = [train_dataset.get_emotion_name(i) for i in range(7)]
        train_values = [train_counts[i] for i in range(7)]
        train_total = sum(train_values)
        
        # 统计测试集分布
        test_counts = Counter(test_dataset.labels)
        test_emotion_names = [test_dataset.get_emotion_name(i) for i in range(7)]
        test_values = [test_counts[i] for i in range(7)]
        test_total = sum(test_values)
        
        print(f"训练集总样本数: {train_total}")
        print(f"测试集总样本数: {test_total}")
        
        # 创建饼图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('FER2013 数据集表情分布', fontsize=20, fontweight='bold')
        
        # 定义颜色方案
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF', '#FFD700']
        
        # 训练集饼图
        wedges1, texts1, autotexts1 = ax1.pie(
            train_values, 
            labels=train_emotion_names,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.05 if count == min(train_values) else 0 for count in train_values],  # 突出最小类别
            shadow=True,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        ax1.set_title(f'训练集表情分布\n总样本数: {train_total:,}', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # 在训练集饼图上添加样本数量
        for i, (text, count) in enumerate(zip(autotexts1, train_values)):
            percentage = float(text.get_text().replace('%', ''))
            if count < 500:  # 对于样本较少的类别，显示具体数量
                text.set_text(f'{percentage:.1f}%\n({count})')
            else:
                text.set_text(f'{percentage:.1f}%')
        
        # 测试集饼图
        wedges2, texts2, autotexts2 = ax2.pie(
            test_values,
            labels=test_emotion_names,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=[0.05 if count == min(test_values) else 0 for count in test_values],  # 突出最小类别
            shadow=True,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        ax2.set_title(f'测试集表情分布\n总样本数: {test_total:,}', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # 在测试集饼图上添加样本数量
        for i, (text, count) in enumerate(zip(autotexts2, test_values)):
            percentage = float(text.get_text().replace('%', ''))
            if count < 500:  # 对于样本较少的类别，显示具体数量
                text.set_text(f'{percentage:.1f}%\n({count})')
            else:
                text.set_text(f'{percentage:.1f}%')
        
        # 添加图例说明
        legend_labels = [f'{name}: {train_counts[i]} (训练) / {test_counts[i]} (测试)' 
                        for i, name in enumerate(train_emotion_names)]
        
        fig.legend(wedges1, legend_labels, loc='center', 
                  bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=10,
                  title='样本数量对比', title_fontsize=12)
        
        plt.tight_layout()
        
        # 保存图片
        output_filename = 'data_distribution_pie_charts.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✅ 饼图已保存为: {output_filename}")
        
        # 显示图表
        plt.show()
        
        # 打印详细统计信息
        print("\n" + "=" * 60)
        print("详细数据分布统计")
        print("=" * 60)
        
        print(f"{'表情类别':<10} {'训练集':<10} {'占比':<8} {'测试集':<10} {'占比':<8}")
        print("-" * 50)
        
        for i, emotion in enumerate(train_emotion_names):
            train_pct = (train_values[i] / train_total) * 100
            test_pct = (test_values[i] / test_total) * 100
            print(f"{emotion:<10} {train_values[i]:<10} {train_pct:<8.1f}% "
                  f"{test_values[i]:<10} {test_pct:<8.1f}%")
        
        print("-" * 50)
        print(f"{'总计':<10} {train_total:<10} {'100.0%':<8} {test_total:<10} {'100.0%':<8}")
        
        # 分析数据不平衡性
        print(f"\n数据平衡性分析:")
        max_train = max(train_values)
        min_train = min(train_values)
        imbalance_ratio = max_train / min_train
        
        print(f"训练集不平衡比率: {imbalance_ratio:.2f}:1")
        print(f"最多类别: {train_emotion_names[train_values.index(max_train)]} ({max_train})")
        print(f"最少类别: {train_emotion_names[train_values.index(min_train)]} ({min_train})")
        
        print("=" * 60)
        
        return {
            'train_distribution': dict(zip(train_emotion_names, train_values)),
            'test_distribution': dict(zip(test_emotion_names, test_values)),
            'train_total': train_total,
            'test_total': test_total,
            'imbalance_ratio': imbalance_ratio
        }
        
    except Exception as e:
        print(f"❌ 生成分布图失败: {e}")
        print("请检查数据目录结构和文件完整性")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='数据加载器测试和可视化')
    parser.add_argument('--visualize', action='store_true', 
                       help='生成数据分布饼图')
    parser.add_argument('--stats', action='store_true', 
                       help='计算数据集统计信息')
    parser.add_argument('--test', action='store_true', 
                       help='测试数据加载器')
    parser.add_argument('--all', action='store_true', 
                       help='执行所有功能')
    parser.add_argument('--data_dir', type=str, default='.', 
                       help='数据集根目录')
    
    args = parser.parse_args()
    
    # 默认执行可视化
    if not any([args.visualize, args.stats, args.test, args.all]):
        args.visualize = True
    
    data_dir = args.data_dir
    
    # 检查数据目录
    if not (os.path.exists(os.path.join(data_dir, 'train')) and os.path.exists(os.path.join(data_dir, 'test'))):
        print("❌ 未找到train和test文件夹")
        print("请确保数据按以下结构组织:")
        print(".")
        print("├── train/")
        print("│   ├── angry/")
        print("│   ├── disgust/")
        print("│   ├── fear/")
        print("│   ├── happy/")
        print("│   ├── neutral/")
        print("│   ├── sad/")
        print("│   └── surprise/")
        print("└── test/")
        print("    ├── angry/")
        print("    ├── disgust/")
        print("    ├── fear/")
        print("    ├── happy/")
        print("    ├── neutral/")
        print("    ├── sad/")
        print("    └── surprise/")
        exit(1)
    
    print("✅ 检测到文件夹格式的数据集")
    
    try:
        if args.visualize or args.all:
            print("\n🎨 生成数据分布可视化...")
            result = visualize_data_distribution(data_dir)
            
        if args.stats or args.all:
            print("\n📊 计算数据集统计信息...")
            calculate_dataset_stats(data_dir)
            
        if args.test or args.all:
            print("\n🧪 测试数据加载器...")
            train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=8)
            
            print(f"训练批次数量: {len(train_loader)}")
            print(f"验证批次数量: {len(val_loader)}")
            print(f"测试批次数量: {len(test_loader)}")
            
            # 测试一个批次
            images, labels = next(iter(train_loader))
            print(f"图像批次形状: {images.shape}")
            print(f"标签批次形状: {labels.shape}")
            print(f"标签值范围: {labels.min().item()} - {labels.max().item()}")
            
        print("\n✅ 所有操作完成!")
        
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        import traceback
        traceback.print_exc()