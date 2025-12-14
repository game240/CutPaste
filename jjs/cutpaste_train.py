from typing import List, Dict, Any, Sequence, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from jjs.cutpaste import make_cutpaste_rect_pair


class CutPasteBinaryDataset(Dataset):
    """정상 이미지 경로들을 받아서 (정상, 사각형 CutPaste) 이진 분류용 샘플을 만드는 Dataset.

    - 길이: 2 * N (N = 정상 이미지 개수)
    - 짝수 인덱스: (원본, label=0)
    - 홀수 인덱스: (CutPaste 증강, label=1)
    """

    def __init__(self, image_paths: Sequence[str], transform):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths) * 2

    def __getitem__(self, idx: int):
        base_idx = idx // 2
        is_aug = idx % 2  # 0: orig, 1: aug

        path = self.image_paths[base_idx]
        orig, aug = make_cutpaste_rect_pair(path, self.transform)

        if is_aug == 0:
            x = orig
            y = 0  # 정상
        else:
            x = aug
            y = 1  # CutPaste 증강

        return x, y


class CutPasteBinaryResNet(nn.Module):
    """ResNet18 기반 이진 분류 모델

    - backbone: global average pooling 후 512-d feature 출력
    - classifier: 512 -> 2 (정상 vs 증강)
    - forward: (logits, features) 반환
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 2):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()  # 512-d feature 출력
        self.backbone = backbone
        self.classifier = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        feats = self.backbone(x)          # (B, 512)
        logits = self.classifier(feats)   # (B, 2)
        return logits, feats


def _train_one_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, feats = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def _evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, feats = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


def train_cutpaste_binary(
    image_paths: Sequence[str],
    transform,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    val_ratio: float = 0.2,
    num_workers: int = 4,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[CutPasteBinaryResNet, nn.Module, Dict[str, Any]]:
    """정상 vs 사각형 CutPaste self-supervised 학습을 수행하고 feature_extractor 를 반환.

    Args:
        image_paths: 정상 이미지 경로 리스트 (여러 카테고리 섞어서 전달 가능)
        transform: 이미지 → 텐서 변환 (예: normal_transform)
        epochs: 학습 epoch 수
        batch_size: 미니배치 크기
        lr: 학습률
        val_ratio: 검증 비율 (0~1, 예: 0.2 → 80/20)
        num_workers: DataLoader num_workers
        pretrained: ImageNet pretrained ResNet 사용 여부
        device: torch.device (None 이면 자동 cuda/cpu 선택)

    Returns:
        model: 학습된 CutPasteBinaryResNet
        feature_extractor: model.backbone.eval() (512-d feature extractor)
        history: epoch별 loss/acc 기록 딕셔너리
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = list(image_paths)
    num_total = len(image_paths)
    num_train = int(num_total * (1.0 - val_ratio))

    train_paths = image_paths[:num_train]
    val_paths = image_paths[num_train:]

    print(f"총 정상 이미지 개수: {num_total}")
    print(f" - train: {len(train_paths)}장, val: {len(val_paths)}장")
    print("사용 디바이스:", device)

    train_dataset = CutPasteBinaryDataset(train_paths, transform)
    val_dataset = CutPasteBinaryDataset(val_paths, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = CutPasteBinaryResNet(pretrained=pretrained, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    feature_extractor = model.backbone.eval()

    print("\n[완료] self-supervised CutPaste (정상 vs 사각형 증강) 학습이 끝났습니다.")
    print("[완료] 512-d feature extractor 준비 완료: 'feature_extractor' 변수를 사용하면 됩니다.")

    return model, feature_extractor, history


__all__ = [
    "CutPasteBinaryDataset",
    "CutPasteBinaryResNet",
    "train_cutpaste_binary",
]




