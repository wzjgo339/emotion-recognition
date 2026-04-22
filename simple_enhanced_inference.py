#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化增强的表情识别推理
避免库版本兼容性问题
"""

import torch
import cv2
import numpy as np
import os
import argparse

from model import CNNWithAttention


class SimpleEnhancedEmotionRecognizer:
    """简化的增强表情识别器"""
    
    def __init__(self, model_path='best_fixed_model.pth', device='auto'):
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载模型
        self.model = CNNWithAttention(num_classes=7).to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型: {model_path}")
        else:
            print(f"警告: 模型文件 {model_path} 不存在，使用未训练的模型")
        
        self.model.eval()
        
        # 表情类别
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # 图像预处理
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5076], std=[0.2128])
        ])
    

    

    
    def enhance_image_quality(self, image):
        """基础图像质量增强"""
        try:
            # 直方图均衡化
            if len(image.shape) == 3:
                # 彩色图像
                yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            else:
                # 灰度图像
                image = cv2.equalizeHist(image)
            
            # 轻微锐化
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            if len(image.shape) == 3:
                image = cv2.filter2D(image, -1, kernel)
            else:
                image = cv2.filter2D(image.reshape(-1, 1), -1, kernel).reshape(image.shape)
            
            return image
        except:
            return image
    

    
    def extract_smart_center(self, image):
        """智能中心截取 - 基于黄金比例"""
        h, w = image.shape[:2]
        
        # 使用黄金比例截取中心区域，减少背景
        # 通常人脸在图像中心偏上的位置
        center_x, center_y = w // 2, h // 2
        
        # 动态计算截取区域大小（适应不同图像尺寸）
        if h > w:  # 竖图
            crop_size = min(w, h * 0.6)  # 60%的高度
        else:  # 横图
            crop_size = min(h, w * 0.7)  # 70%的宽度
        
        crop_size = int(crop_size)
        
        # 计算截取坐标（稍微偏上，符合人脸位置习惯）
        x1 = max(0, center_x - crop_size // 2)
        y1 = max(0, center_y - crop_size // 2 - crop_size // 8)  # 向上偏移1/8
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        
        return image[y1:y2, x1:x2]
    
    def extract_rule_based(self, image):
        """基于规则的截取 - 简单但有效"""
        h, w = image.shape[:2]
        
        # 转换为灰度图分析亮度分布
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 简单边缘检测找到可能的人脸区域
        edges = cv2.Canny(gray, 50, 150)
        
        # 计算水平和垂直投影
        h_proj = np.sum(edges, axis=1)
        v_proj = np.sum(edges, axis=0)
        
        # 找到边缘密度最大的区域（可能是人脸）
        h_peak = np.argmax(h_proj[:h//2]) + h//4  # 上半部分
        v_peak = np.argmax(v_proj)  # 水平中心附近
        
        # 以峰值为中心截取区域
        crop_size = min(h, w) // 2
        x1 = max(0, v_peak - crop_size // 2)
        y1 = max(0, h_peak - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        
        return image[y1:y2, x1:x2]
    
    def extract_center_region(self, image):
        """简单中心截取"""
        h, w = image.shape[:2]
        
        # 截取中心2/3区域
        x1 = w // 6
        y1 = h // 6
        x2 = w - w // 6
        y2 = h - h // 6
        
        return image[y1:y2, x1:x2]
    
    def preprocess_image_enhanced(self, image_path, method='no_crop'):
        """增强图像预处理 - 简化版本（无人脸检测）"""
        # 1. 读取图像 - 支持中文路径
        try:
            # 方法1：使用cv2.imdecode从字节读取
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                # 方法2：如果失败，尝试直接读取
                image = cv2.imread(image_path)
                
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
        except Exception as e:
            raise ValueError(f"读取图像失败: {image_path}, 错误: {str(e)}")
        
        # 2. 根据方法选择截取策略
        if method == 'no_crop':
            # 不截取，直接使用整张图片
            processed_region = image
            
        elif method == 'smart_center':
            # 智能中心截取 - 更简单稳定
            processed_region = self.extract_smart_center(image)
            
        elif method == 'rule_based':
            # 基于规则的截取 - 结合图像特征
            processed_region = self.extract_rule_based(image)
            
        else:
            # 默认中心截取
            processed_region = self.extract_center_region(image)
        
        # 3. 调整大小为48x48
        resized = cv2.resize(processed_region, (48, 48))
        
        # 4. 转为灰度图
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
        
        return gray, image
    
    def visualize_prediction(self, image_path, predicted_emotion, confidence, probabilities, processed_image, output_dir=None):
        """可视化预测结果"""
        import matplotlib.pyplot as plt
        import matplotlib
        
        # 设置中文字体
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
            
        # 读取原始图像
        with open(image_path, 'rb') as f:
            image_data = f.read()
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if original_image is None:
            print("[警告] 无法读取原始图像进行可视化")
            return
        
        # 创建图表 - 简化为3个子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'表情识别结果: {predicted_emotion} ({confidence:.1%})', fontsize=16)
        
        # 1. 原始图像
        ax1 = axes[0]
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('原始图像')
        ax1.axis('off')
        
        # 2. 处理后的图像
        ax2 = axes[1]
        ax2.imshow(processed_image, cmap='gray')
        ax2.set_title('处理后的图像 (48x48)')
        ax2.axis('off')
        
        # 3. 概率分布柱状图
        ax3 = axes[2]
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        probs = [prob.item() for prob in probabilities]
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan']
        
        bars = ax3.bar(emotions, probs, color=colors)
        ax3.set_title('表情概率分布')
        ax3.set_ylabel('概率')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        # 高亮预测结果
        pred_idx = emotions.index(predicted_emotion)
        bars[pred_idx].set_edgecolor('black')
        bars[pred_idx].set_linewidth(3)
        
        # 在柱子上显示百分比
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{prob:.1%}', ha='center', va='bottom', fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
        else:
            # 默认保存到当前工作目录下的 inference_results 文件夹
            output_dir = os.path.join(os.getcwd(), "inference_results")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[可视化] 结果已保存: {output_path}")
        
        # 显示图像（如果在支持的环境中）
        try:
            plt.show()
        except:
            print("[提示] 请查看保存的可视化图像文件")
        
        plt.close()
    
    def preprocess_image_simple(self, image_path):
        """简单图像预处理（用于对比）"""
        # 1. 读取为灰度图 - 支持中文路径
        try:
            # 先尝试彩色读取再转灰度
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                image = cv2.imread(image_path)
                
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
                
            # 转为灰度图
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
        except Exception as e:
            raise ValueError(f"读取图像失败: {image_path}, 错误: {str(e)}")
        
        # 2. 直接调整大小
        resized = cv2.resize(image, (48, 48))
        
        return resized
    
    def predict_enhanced(self, image_path, return_probabilities=False, method='no_crop', visualize=False, output_dir=None):
        """使用增强预处理预测"""
        try:
            # 增强预处理
            processed_image, original_image = self.preprocess_image_enhanced(image_path, method)
            
            # 转换为tensor
            image_tensor = self.transform(processed_image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_emotion = self.emotions[predicted_class]
            
            # 可视化结果
            if visualize:
                self.visualize_prediction(image_path, predicted_emotion, confidence, 
                                        probabilities[0], processed_image, output_dir)
            
            if return_probabilities:
                probs_dict = {emotion: prob.item() 
                            for emotion, prob in zip(self.emotions, probabilities[0])}
                return predicted_emotion, confidence, probs_dict
            else:
                return predicted_emotion, confidence
                
        except Exception as e:
            print(f"增强预处理预测失败: {e}")
            return None, 0.0
    
    def predict_simple(self, image_path, return_probabilities=False):
        """使用简单预处理预测"""
        try:
            # 简单预处理
            face_image = self.preprocess_image_simple(image_path)
            
            # 转换为tensor
            image_tensor = self.transform(face_image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_emotion = self.emotions[predicted_class]
            
            if return_probabilities:
                probs_dict = {emotion: prob.item() 
                            for emotion, prob in zip(self.emotions, probabilities[0])}
                return predicted_emotion, confidence, probs_dict
            else:
                return predicted_emotion, confidence
                
        except Exception as e:
            print(f"简单预处理预测失败: {e}")
            return None, 0.0
    
    def compare_methods(self, image_path):
        """比较两种预处理方法"""
        print(f"比较预处理方法: {image_path}")
        print("=" * 60)
        
        try:
            # 简单预处理预测
            simple_emotion, simple_confidence, simple_probs = self.predict_simple(image_path, return_probabilities=True)
            
            # 增强预处理预测
            enhanced_emotion, enhanced_confidence, enhanced_probs = self.predict_enhanced(
                image_path, return_probabilities=True
            )
            
            # 显示结果
            print(f"简单预处理:")
            if simple_emotion:
                print(f"  预测表情: {simple_emotion}")
                print(f"  置信度: {simple_confidence:.1%}")
                print(f"  概率分布:")
                for emotion, prob in simple_probs.items():
                    print(f"    {emotion}: {prob:.3f}")
            else:
                print("  预测失败")
            print()
            
            print(f"增强预处理:")
            if enhanced_emotion:
                print(f"  预测表情: {enhanced_emotion}")
                print(f"  置信度: {enhanced_confidence:.1%}")
                print(f"  概率分布:")
                for emotion, prob in enhanced_probs.items():
                    print(f"    {emotion}: {prob:.3f}")
            else:
                print("  预测失败")
            print()
            
            # 比较结果
            if simple_emotion and enhanced_emotion:
                if simple_emotion == enhanced_emotion:
                    print("✅ 预测结果相同")
                    
                    # 比较置信度
                    if enhanced_confidence > simple_confidence + 0.05:
                        print(f"📈 增强预处理置信度显著提升 (+{enhanced_confidence-simple_confidence:.1%})")
                    elif simple_confidence > enhanced_confidence + 0.05:
                        print(f"📉 简单预处理置信度更高 (+{simple_confidence-enhanced_confidence:.1%})")
                    else:
                        print("🤝 置信度相近")
                else:
                    print("⚠️ 预测结果不同！预处理方法显著影响结果")
                    
                    # 显示概率差异
                    print("概率差异:")
                    for emotion in self.emotions:
                        diff = enhanced_probs[emotion] - simple_probs[emotion]
                        if abs(diff) > 0.1:
                            print(f"  {emotion}: {diff:+.3f}")
            else:
                print("❌ 无法比较（预测失败）")
            
            return {
                'simple_emotion': simple_emotion,
                'simple_confidence': simple_confidence,
                'enhanced_emotion': enhanced_emotion,
                'enhanced_confidence': enhanced_confidence,
                'same_prediction': simple_emotion == enhanced_emotion if simple_emotion and enhanced_emotion else None
            }
            
        except Exception as e:
            print(f"比较失败: {e}")
            return None
    
    def save_enhanced_face(self, image_path, output_path='enhanced_face.jpg'):
        """保存增强预处理后的图像"""
        try:
            processed_image, original_image = self.preprocess_image_enhanced(image_path)
            
            # 保存处理后的48x48图像
            cv2.imwrite(output_path, processed_image)
            print(f"[保存] 增强预处理后的图像已保存: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"保存失败: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='简化的增强表情识别推理')
    parser.add_argument('--model', type=str, default='best_fixed_model.pth', help='模型文件路径')
    parser.add_argument('--image', type=str, help='图像文件路径')
    parser.add_argument('--batch', type=str, help='批量处理文件夹路径')
    parser.add_argument('--webcam', action='store_true', help='实时摄像头识别')
    parser.add_argument('--compare', action='store_true', help='比较预处理方法')
    parser.add_argument('--save_face', action='store_true', help='保存处理后的图像')
    parser.add_argument('--visualize', action='store_true', help='启用可视化结果')
    parser.add_argument('--device', type=str, default='auto', help='设备选择')
    parser.add_argument('--method', type=str, default='no_crop',
                       choices=['no_crop', 'smart_center', 'rule_based', 'center_region'],
                       help='选择截取方法')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.image and not args.batch and not args.webcam:
        print("请提供 --image, --batch 或 --webcam 参数")
        print("示例:")
        print("  python simple_enhanced_inference.py --image angry.jpg")
        print("  python simple_enhanced_inference.py --image angry.jpg --compare")
        print("  python simple_enhanced_inference.py --batch ./photos/")
        print("  python simple_enhanced_inference.py --batch ./photos/ --save_face")
        print("  python simple_enhanced_inference.py --webcam")
        return
    
    # 创建识别器
    recognizer = SimpleEnhancedEmotionRecognizer(args.model, args.device)
    
    if args.webcam:
        # 摄像头实时识别模式
        process_webcam(recognizer)
    elif args.batch:
        # 批量处理模式
        process_batch(recognizer, args.batch, args.save_face, args.visualize)
    else:
        # 单张图像处理模式
        # 检查图像是否存在
        if not os.path.exists(args.image):
            print(f"图像文件不存在: {args.image}")
            return
        
        # 执行预测
        if args.compare:
            result = recognizer.compare_methods(args.image)
            
            # 保存处理后的人脸
            if args.save_face:
                recognizer.save_enhanced_face(args.image)
        else:
            # 只使用增强预处理
            emotion, confidence = recognizer.predict_enhanced(
                args.image, method=args.method, visualize=args.visualize
            )
            
            if emotion:
                print(f"预测结果 (方法: {args.method}):")
                print(f"  预测表情: {emotion}")
                print(f"  置信度: {confidence:.1%}")
                
                # 保存处理后的图像
                if args.save_face:
                    recognizer.save_enhanced_face(args.image)
            else:
                print("预测失败")


def process_batch(recognizer, folder_path, save_faces=False, visualize=False):
    """批量处理文件夹中的图像"""
    import glob
    from pathlib import Path
    
    # 创建结果文件夹
    results_folder = os.path.join(folder_path, "results")
    os.makedirs(results_folder, exist_ok=True)
    
    print(f"开始批量处理文件夹: {folder_path}")
    print(f"结果将保存在: {results_folder}")
    print("="*60)
    
    if not os.path.isdir(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return
    
    # 支持的图像格式（避免重复，Windows系统不区分大小写）
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # 去重处理（防止同一文件被多次匹配）
    image_files = list(set(image_files))
    
    if not image_files:
        print(f"❌ 在文件夹中未找到图像文件: {folder_path}")
        return
    
    print(f"发现 {len(image_files)} 张图像")
    print()
    
    # 统计结果
    results = {
        'total': len(image_files),
        'success': 0,
        'failed': 0,
        'emotions': {}
    }
    
    # 处理每张图像
    for i, image_path in enumerate(image_files, 1):
        print(f"处理 {i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            emotion, confidence = recognizer.predict_enhanced(image_path, method='no_crop', visualize=visualize, output_dir=results_folder)
            
            if emotion:
                results['success'] += 1
                
                # 统计表情
                if emotion not in results['emotions']:
                    results['emotions'][emotion] = 0
                results['emotions'][emotion] += 1
                
                # 详细显示单张图片结果
                print(f"  预测结果: {emotion} (置信度: {confidence:.1%})")
                
                # 单独显示可视化结果（如果启用了可视化）
                if visualize:
                    print(f"  可视化图已保存到: {results_folder}")
                    
                    # 短暂暂停以便查看结果
                    import time
                    time.sleep(0.5)
                
                # 保存处理后的图像
                if save_faces:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    face_output = os.path.join(results_folder, f"enhanced_{base_name}.jpg")
                    recognizer.save_enhanced_face(image_path, face_output)
                    print(f"  处理后图像已保存: {face_output}")
                
                print()  # 空行分隔不同图片的结果
            else:
                results['failed'] += 1
                print(f"  识别失败")
                print()
                
        except Exception as e:
            results['failed'] += 1
            print(f"  处理出错: {str(e)}")
            print()
    
    # 打印统计结果
    print()
    print("="*60)
    print("批量处理统计")
    print("="*60)
    print(f"总处理数量: {results['total']}")
    print(f"成功识别: {results['success']} ({results['success']/results['total']*100:.1f}%)")
    print(f"识别失败: {results['failed']} ({results['failed']/results['total']*100:.1f}%)")
    


    
    print("="*60)

def process_webcam(recognizer):
    """摄像头实时识别"""
    print("🎮 启动实时摄像头识别...")
    print("按 'q' 键退出，按 's' 键截图保存")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("✅ 摄像头已打开，开始识别...")
    
    # 调整摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    save_count = 0
    
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            frame_count += 1
            
            # 每隔几帧进行一次识别（提高性能）
            if frame_count % 3 == 0:
                # 转换为RGB用于显示
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # 临时保存当前帧
                    temp_path = "temp_webcam_frame.jpg"
                    cv2.imwrite(temp_path, frame)
                    
                    # 进行表情识别
                    emotion, confidence = recognizer.predict_enhanced(temp_path)
                    
                    # 删除临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # 在画面上显示结果
                    if emotion:
                        # 绘制结果文字
                        text = f"{emotion}: {confidence:.1%}"
                        cv2.putText(display_frame, text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "Recognition failed", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                except Exception as e:
                    print(f"识别错误: {e}")
                    cv2.putText(display_frame, "Recognition Error", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 转换回BGR用于OpenCV显示
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            
            # 显示画面
            cv2.imshow('Emotion Recognition', display_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 用户退出")
                break
            elif key == ord('s'):
                # 保存截图
                save_count += 1
                save_path = f"webcam_capture_{save_count}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"📸 截图已保存: {save_path}")
    
    except KeyboardInterrupt:
        print("\n👋 用户中断")
    
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 摄像头已关闭")


if __name__ == "__main__":
    main()