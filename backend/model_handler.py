import sys
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import CNNWithAttention


class ModelHandler:
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def __init__(self, model_path='best_model.pth', device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.model_path = model_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5076], std=[0.2128])
        ])

    def load(self):
        self.model = CNNWithAttention(num_classes=7).to(self.device)
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model.eval()
        return self

    def preprocess(self, image_bytes):
        """Convert uploaded image bytes -> (1, 1, 48, 48) tensor."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode image")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))

        tensor = self.transform(resized)
        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image_bytes):
        """Returns (emotion: str, confidence: float, probabilities: dict)."""
        input_tensor = self.preprocess(image_bytes)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()

        probabilities = {
            emotion: round(probs[0][i].item(), 4)
            for i, emotion in enumerate(self.EMOTIONS)
        }

        return self.EMOTIONS[pred_idx], round(confidence, 4), probabilities


_model_handler = None


def get_model_handler(model_path='best_model.pth'):
    global _model_handler
    if _model_handler is None:
        _model_handler = ModelHandler(model_path).load()
    return _model_handler
