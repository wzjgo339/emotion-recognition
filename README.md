# 表情识别系统 (Emotion Recognition)

基于深度学习的七类面部表情识别：`angry` · `disgust` · `fear` · `happy` · `sad` · `surprise` · `neutral`

## 模型架构

| 组件 | 说明 |
|------|------|
| Backbone | 4 层卷积 (1→64→128→256→512) |
| Attention | SelfAttention + SEBlock 通道注意力 |
| Classifier | 全局平均池化 + 3 层 FC (512→256→128→7) |
| 输入 | 48×48 灰度图像 |
| 参数量 | ~4.2M |
| 准确率 | 68% (FER2013 test set) |

## 快速开始

### CLI 模式

```bash
pip install -r requirements.txt

# 训练
python train.py --data_dir . --epochs 50

# 单张图片识别
python simple_enhanced_inference.py --image path/to/image.jpg

# 摄像头实时识别
python simple_enhanced_inference.py --webcam

# 模型评估
python evaluate_model.py --model best_model.pth --test_dir ./test
```

### Web 模式

```
后端：FastAPI + PyTorch  (http://localhost:8000)
前端：React + Vite      (http://localhost:5173)
```

**启动后端**（需要安装 PyTorch 的环境）：
```bash
cd backend
pip install -r requirements.txt
# 从项目根目录启动
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**启动前端**（新开终端）：
```bash
cd frontend
npm install
npm run dev
```

打开 `http://localhost:5173`，上传图片即可识别。

## 技术栈

CLI: PyTorch · OpenCV · NumPy · Matplotlib · scikit-learn

Web: FastAPI · React · Vite · TailwindCSS · Recharts
