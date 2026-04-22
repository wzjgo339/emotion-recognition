# 表情识别项目 (Emotion Recognition)

基于深度学习的面部表情识别系统。

## 项目结构

```
├── train/              # 训练数据集
├── test/               # 测试数据集
├── model.py           # 模型定义
├── train.py           # 训练脚本
├── inference.py       # 推理脚本
├── data_loader.py     # 数据加载
├── requirements.txt   # 依赖包
└── best_model.pth     # 训练好的模型权重
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型
```bash
python train.py
```

### 推理预测
```bash
python inference.py
```

## 技术栈

- Python 3.x
- PyTorch
- OpenCV
- Haar Cascade
