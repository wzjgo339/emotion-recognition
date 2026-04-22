#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的模型评估脚本
评估best_fixed_model.pth在测试集上的性能
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import defaultdict
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from datetime import datetime

# 导入项目模块
from model import CNNWithAttention
from data_loader import FER2013Dataset

class ModelEvaluator:
    def __init__(self, model_path):
        """初始化评估器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # 加载模型
        self.model = CNNWithAttention(num_classes=7).to(self.device)
        self.load_model()
        
        print(f"[成功] 模型加载成功: {model_path}")
        print(f"[设备] 使用设备: {self.device}")
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except Exception as e:
            print(f"[错误] 模型加载失败: {e}")
            raise
    
    def load_test_data(self, test_dir='test', batch_size=32):
        """加载测试数据集"""
        try:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5076], std=[0.2128])
            ])
            
            test_dataset = FER2013Dataset('.', transform=transform, mode='test')  # 从当前目录加载，让data_loader自己找test目录
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            print(f"[数据] 测试集大小: {len(test_dataset)} 张图片")
            return test_loader, len(test_dataset)
            
        except Exception as e:
            print(f"[错误] 测试数据加载失败: {e}")
            raise
    
    def evaluate(self, test_loader):
        """评估模型性能"""
        print("[评估] 开始评估...")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        incorrect_predictions = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # 记录错误预测的样本
                for i in range(len(predictions)):
                    if predictions[i] != labels[i]:
                        incorrect_predictions.append({
                            'true_label': labels[i].item(),
                            'predicted_label': predictions[i].item(),
                            'confidence': probabilities[i][predictions[i]].item(),
                        })
                
                # 显示进度
                if (batch_idx + 1) % 10 == 0:
                    print(f"  已处理: {(batch_idx + 1) * len(images)}/{len(test_loader.dataset)} 张图片")
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities), incorrect_predictions
    
    def generate_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """生成混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(self.emotions))
        plt.xticks(tick_marks, self.emotions, rotation=45)
        plt.yticks(tick_marks, self.emotions)
        
        # 在单元格中添加数字
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[保存] 混淆矩阵已保存: {save_path}")
        return cm
    
    def generate_class_report(self, y_true, y_pred, save_path='classification_report.txt'):
        """生成分类报告"""
        report = classification_report(y_true, y_pred, target_names=self.emotions, output_dict=True)
        report_text = classification_report(y_true, y_pred, target_names=self.emotions)
        
        # 保存文本报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Model Classification Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n\n")
            f.write(report_text)
            f.write("\n" + "="*60 + "\n")
        
        print(f"[保存] 分类报告已保存: {save_path}")
        return report
    
    def run_evaluation(self, test_dir='test', save_dir='evaluation_results'):
        """运行完整评估流程"""
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        print("[开始] 开始模型评估...")
        print(f"[数据] 测试数据目录: {test_dir}")
        print(f"[输出] 结果保存目录: {save_dir}")
        print("="*60)
        
        # 1. 加载测试数据
        test_loader, test_size = self.load_test_data(test_dir)
        
        # 2. 评估模型
        y_pred, y_true, probabilities, incorrect_predictions = self.evaluate(test_loader)
        
        # 3. 计算准确率
        total_samples = len(y_true)
        correct_predictions = np.sum(y_true == y_pred)
        accuracy = correct_predictions / total_samples
        
        # 4. 生成可视化报告
        print("\n[生成] 生成可视化报告...")
        
        # 混淆矩阵
        cm = self.generate_confusion_matrix(y_true, y_pred, 
                                           os.path.join(save_dir, 'confusion_matrix.png'))
        
        # 分类报告
        report = self.generate_class_report(y_true, y_pred,
                                          os.path.join(save_dir, 'classification_report.txt'))
        
        # 5. 生成综合报告
        print("\n[报告] 生成综合评估报告...")
        
        # 计算各表情的准确率
        class_accuracies = {}
        for i, emotion in enumerate(self.emotions):
            class_mask = y_true == i
            if np.sum(class_mask) > 0:
                class_acc = np.sum((y_true == y_pred) & class_mask) / np.sum(class_mask)
                class_accuracies[emotion] = class_acc
        
        # 生成报告内容
        report_content = f"""
{'='*80}
                        Model Evaluation Report
{'='*80}

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {self.model_path}
Device: {self.device}

Overall Performance:
────────────────────────────────────────────────────────────────
• Total Samples: {total_samples:,}
• Correct Predictions: {correct_predictions:,}
• Wrong Predictions: {total_samples - correct_predictions:,}
• Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)

Per-Class Performance:
────────────────────────────────────────────────────────────────
"""
        
        for emotion in self.emotions:
            if emotion in report:
                metrics = report[emotion]
                report_content += f"""
{emotion.upper():12}: Precision {metrics['precision']:.3f} | Recall {metrics['recall']:.3f} | F1 {metrics['f1-score']:.3f} | Support {int(metrics['support'])}
"""
        
        report_content += f"""

Per-Class Accuracy:
────────────────────────────────────────────────────────────────
"""
        for emotion, acc in class_accuracies.items():
            report_content += f"• {emotion:10}: {acc:.4f} ({acc*100:.2f}%)\n"
        
        if incorrect_predictions:
            report_content += f"""

Error Analysis:
────────────────────────────────────────────────────────────────
• Total Errors: {len(incorrect_predictions)}
• Error Rate: {(1-accuracy)*100:.2f}%
• Average Error Confidence: {np.mean([pred['confidence'] for pred in incorrect_predictions]):.3f}
"""
        
        report_content += f"""

Generated Files:
────────────────────────────────────────────────────────────────
• Confusion Matrix: {save_dir}/confusion_matrix.png
• Classification Report: {save_dir}/classification_report.txt
• Summary Report: {save_dir}/evaluation_summary.txt

{'='*80}
"""
        
        # 保存报告
        summary_path = os.path.join(save_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"[保存] 综合评估报告已保存: {summary_path}")
        print("\n[报告内容预览]")
        print("="*60)
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Total Samples: {len(y_true)}")
        print(f"Correct Predictions: {np.sum(y_true == y_pred)}")
        print(f"Wrong Predictions: {len(y_true) - np.sum(y_true == y_pred)}")
        print("\nPer-Class Accuracy:")
        for emotion, acc in class_accuracies.items():
            print(f"  {emotion}: {acc:.4f} ({acc*100:.2f}%)")
        
        print(f"\n[完成] 评估完成！")
        print(f"[结果] 总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"[输出] 所有结果已保存到: {save_dir}")
        
        return accuracy, class_accuracies


def main():
    parser = argparse.ArgumentParser(description='Emotion Recognition Model Evaluator')
    parser.add_argument('--model', type=str, default='best_model.pth',
                       help='Model file path')
    parser.add_argument('--test_dir', type=str, default='test',
                       help='Test dataset directory')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Results save directory')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"[错误] 模型文件不存在: {args.model}")
        return
    
    # 检查测试目录是否存在
    if not os.path.exists(args.test_dir):
        print(f"[错误] 测试目录不存在: {args.test_dir}")
        return
    
    try:
        # 创建评估器并运行评估
        evaluator = ModelEvaluator(args.model)
        accuracy, class_accuracies = evaluator.run_evaluation(args.test_dir, args.save_dir)
        
        print("\n[成功] 评估成功完成！")
        
    except Exception as e:
        print(f"[错误] 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()