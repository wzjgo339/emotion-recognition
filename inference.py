import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

from model import CNNWithAttention


class EmotionRecognizer:
    """表情识别器"""
    def __init__(self, model_path='best_model.pth', device='auto'):
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

    def preprocess_image(self, image):
        """预处理图像"""
        if isinstance(image, str):
            # 从文件路径加载
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"无法加载图像: {image}")
        
        if isinstance(image, np.ndarray):
            # 如果是彩色图像，转为灰度
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 调整大小为48x48
            image = cv2.resize(image, (48, 48))
        
        elif isinstance(image, Image.Image):
            # PIL图像
            if image.mode != 'L':
                image = image.convert('L')
            image = image.resize((48, 48))
            image = np.array(image)
        
        # 应用预处理变换
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor

    def predict(self, image, return_probabilities=False):
        """预测表情"""
        try:
            # 预处理图像
            input_tensor = self.preprocess_image(image)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
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
            print(f"预测失败: {e}")
            return None, 0.0

    def predict_batch(self, images):
        """批量预测"""
        results = []
        for image in images:
            emotion, confidence = self.predict(image)
            results.append((emotion, confidence))
        return results

    def visualize_prediction(self, image):
        """可视化预测结果"""
        # 获取预测结果
        emotion, confidence, probs = self.predict(image, return_probabilities=True)
        
        if emotion is None:
            print("无法预测表情")
            return
        
        # 准备图像显示
        if isinstance(image, str):
            img_display = cv2.imread(image)
            img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                img_display = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                img_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_display = np.array(image)
        
        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 显示原始图像
        ax1.imshow(img_display)
        ax1.set_title(f'预测: {emotion} ({confidence:.2%})')
        ax1.axis('off')
        
        # 显示概率分布
        emotions_list = list(probs.keys())
        probs_list = list(probs.values())
        
        bars = ax2.bar(emotions_list, probs_list, color='skyblue')
        ax2.set_title('各表情概率分布')
        ax2.set_ylabel('概率')
        ax2.set_ylim(0, 1)
        
        # 在柱状图上添加数值
        for bar, prob in zip(bars, probs_list):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        # 高亮预测结果
        pred_idx = emotions_list.index(emotion)
        bars[pred_idx].set_color('orange')
        
        plt.tight_layout()
        plt.show()

    def real_time_detection(self):
        """实时摄像头表情检测"""
        print("开始实时表情检测，按 'q' 退出...")
        
        # 初始化摄像头
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        # 加载人脸检测器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # 提取人脸区域
                face_roi = gray[y:y+h, x:x+w]
                
                # 预测表情
                emotion, confidence = self.predict(face_roi)
                
                if emotion is not None:
                    # 绘制人脸框
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # 显示预测结果
                    label = f'{emotion}: {confidence:.2%}'
                    cv2.putText(frame, label, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow('表情识别', frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='表情识别推理')
    parser.add_argument('--model', type=str, default='best_model.pth', help='模型文件路径')
    parser.add_argument('--image', type=str, help='图像文件路径')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cuda/cpu/auto)')
    parser.add_argument('--webcam', action='store_true', help='使用摄像头实时检测')
    
    args = parser.parse_args()
    
    # 创建识别器
    recognizer = EmotionRecognizer(args.model, args.device)
    
    if args.webcam:
        # 实时检测
        recognizer.real_time_detection()
    elif args.image:
        # 单张图像预测
        if not os.path.exists(args.image):
            print(f"图像文件不存在: {args.image}")
            return
        
        emotion, confidence = recognizer.predict(args.image)
        if emotion:
            print(f"预测表情: {emotion}")
            print(f"置信度: {confidence:.2%}")
        
        # 可视化
        recognizer.visualize_prediction(args.image)
    else:
        print("请提供 --image 参数或使用 --webcam 进行实时检测")
        print("示例:")
        print("  python inference.py --image test.jpg")
        print("  python inference.py --webcam")


if __name__ == "__main__":
    main()