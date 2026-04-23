# 表情识别系统 (Emotion Recognition)

基于深度学习的七类面部表情识别：愤怒、厌恶、恐惧、快乐、悲伤、惊讶、中性。

## 模型架构

- **网络结构**：CNN + SelfAttention + SEBlock
- **输入尺寸**：48×48 灰度图像
- **参数量**：约 4.2M
- **准确率**：68%

## 项目结构

```
.
├── train/                      # 训练数据
├── test/                       # 测试数据
├── model.py                    # 模型定义
├── train.py                    # 训练脚本
├── inference.py                # 基础推理
├── simple_emotion_recognition.py  # 交互界面
├── simple_enhanced_inference.py   # 增强推理
├── evaluate_model.py           # 模型评估
├── show_results.py             # 结果展示
├── quick_fix.py                # 优化训练
├── data_loader.py              # 数据加载
└── requirements.txt            # 依赖
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2.训练模型

```bash
# 基础训练
python train.py

# 针对disgust/fear表情优化训练
python quick_fix.py
```

### 3. 表情识别

```bash
# 交互界面
python simple_emotion_recognition.py

# 单张图片
python inference.py --image path/to/image.jpg

# 批量处理
python simple_enhanced_inference.py
```

### 4. 模型评估
```bash
python evaluate_model.py
python show_results.py
```

## 技术栈

PyTorch | OpenCV | NumPy | PIL
