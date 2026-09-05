"""
Простой скрипт для классификации лица на реальное или сгенерированное.
Использует обученную модель EfficientNet с частотным анализом.

Запуск:
    python inference.py --image_path face.jpg --weights best_model.pth --threshold 0.9
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import math
from scipy import fftpack


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        se_ch = max(1, int(in_ch * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, se_ch),
            Swish(),
            nn.Linear(se_ch, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def drop_connect(x, drop_rate, training):
    if not training or drop_rate == 0.0:
        return x
    keep_prob = 1.0 - drop_rate
    mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep_prob
    return x / keep_prob * mask

class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride, expand_ratio, se_ratio, drop_rate):
        super().__init__()
        self.use_residual = (in_ch == out_ch and stride == 1)
        hidden_ch = in_ch * expand_ratio
        self.drop_rate = drop_rate

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm2d(hidden_ch, eps=1e-3, momentum=0.01),
                Swish()
            ]

        layers += [
            nn.Conv2d(hidden_ch, hidden_ch, k, stride=stride, padding=k//2, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch, eps=1e-3, momentum=0.01),
            Swish()
        ]

        layers.append(SEBlock(hidden_ch, se_ratio))

        layers += [
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01)
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = x + drop_connect(out, self.drop_rate, self.training)
        return out

class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes=1, dropout=0.3):
        super().__init__()

        cfg = [
            (1, 3, 16, 1, 1),
            (6, 3, 24, 2, 2),
            (6, 5, 40, 2, 2),
            (6, 3, 80, 3, 2),
            (6, 5, 112, 3, 1),
            (6, 5, 192, 4, 2),
            (6, 3, 320, 1, 1),
        ]

        width_mult = 1.1
        depth_mult = 1.1

        def round_ch(ch):
            ch *= width_mult
            return max(8, int(ch + 4) // 8 * 8)

        def round_rep(r):
            return int(math.ceil(r * depth_mult))

        stem_ch = round_ch(32)
        self.stem = nn.Sequential(
            nn.Conv2d(4, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch, eps=1e-3, momentum=0.01),
            Swish()
        )

        blocks = []
        in_ch = stem_ch
        total_blocks = sum(round_rep(r) for _, _, _, r, _ in cfg)
        idx = 0

        for exp, k, out, reps, stride in cfg:
            out_ch = round_ch(out)
            for i in range(round_rep(reps)):
                s = stride if i == 0 else 1
                drop = 0.2 * idx / total_blocks
                blocks.append(
                    MBConvBlock(
                        in_ch, out_ch, k, s,
                        expand_ratio=exp,
                        se_ratio=0.25,
                        drop_rate=drop
                    )
                )
                in_ch = out_ch
                idx += 1

        self.blocks = nn.Sequential(*blocks)

        head_ch = round_ch(1280)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch, eps=1e-3, momentum=0.01),
            Swish()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(head_ch, num_classes)
        )

    def forward(self, x, freq_channel=None):
        if freq_channel is not None:
            x = torch.cat([x, freq_channel], dim=1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return self.classifier(x)


def compute_frequency_channel(image_tensor):
    batch_size, channels, height, width = image_tensor.shape
    freq_channels = []

    for i in range(batch_size):
        img_gray = image_tensor[i, 1, :, :].cpu().numpy()
        fft = fftpack.fft2(img_gray)
        fft_shifted = fftpack.fftshift(fft)
        magnitude = np.log(np.abs(fft_shifted) + 1)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        freq_tensor = torch.from_numpy(magnitude).float().unsqueeze(0)
        freq_channels.append(freq_tensor)

    return torch.stack(freq_channels).to(image_tensor.device)


def predict_face(image_path, weights_path, threshold=0.9, device='cuda'):
    """
    Классифицирует лицо на реальное или сгенерированное.
    
    Аргументы:
        image_path (str): путь к изображению
        weights_path (str): путь к файлу весов (.pth)
        threshold (float): порог классификации (по умолчанию 0.9)
        device (str): 'cuda' или 'cpu'
    
    Возвращает:
        pred (int): 1 — фейк, 0 — реальное
        prob (float): вероятность фейка (0..1)
    """
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = EfficientNetCustom(num_classes=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    freq = compute_frequency_channel(img_tensor)

    with torch.no_grad():
        output = model(img_tensor, freq)
        prob = torch.sigmoid(output).item()

    pred = 1 if prob > threshold else 0
    return pred, prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, help='Путь к изображению')
    parser.add_argument('--weights', default='best_model.pth', help='Путь к весам')
    parser.add_argument('--threshold', type=float, default=0.9, help='Порог классификации')
    parser.add_argument('--device', default='cuda', help='cuda или cpu')
    args = parser.parse_args()

    pred, prob = predict_face(args.image_path, args.weights, args.threshold, args.device)
    label = 'Fake' if pred == 1 else 'Real'
    print(f"Изображение: {args.image_path}")
    print(f"Класс: {label}")
    print(f"Вероятность фейка: {prob:.4f}")
