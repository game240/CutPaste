from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
from itertools import chain
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from jjs.data_utils import load_image_as_tensor


class NormalImageDataset(Dataset):
    """정상 이미지 경로 리스트를 받아 텐서로 로드하는 Dataset."""

    def __init__(self, image_paths: Sequence[str], transform):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        return load_image_as_tensor(path, transform=self.transform)


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """임의의 모델과 DataLoader 에 대해 feature (N, D)를 추출."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    feats = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            f = model(x)  # (B, D) 또는 (B, D, ...)
            if isinstance(f, (tuple, list)):
                f = f[0]
            feats.append(f.detach().cpu())
    feats = torch.cat(feats, dim=0).numpy()
    return feats


def fit_gaussian(
    features: np.ndarray,
    eps: float = 1e-5,
) -> Dict[str, np.ndarray]:
    """feature 행렬(N, D)에 대해 Gaussian(μ, Σ)을 학습."""
    mu = features.mean(axis=0)
    cov = np.cov(features, rowvar=False)
    cov += eps * np.eye(cov.shape[0])
    return {"mu": mu, "cov": cov}


def compute_image_gaussians(
    feature_extractor: nn.Module,
    normal_image_paths: Dict[str, List[str]],
    categories: Sequence[str],
    transform,
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
    pretrained_plain: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    nn.Module,
]:
    """정상 이미지를 이용해 CutPaste / Plain ResNet Gaussian 을 각각 학습.

    Args:
        feature_extractor: CutPaste self-supervised 로 학습된 backbone (512-d feature)
        normal_image_paths: 카테고리별 정상 이미지 경로 딕셔너리
        categories: 사용할 카테고리 리스트
        transform: 이미지 → 텐서 변환 (예: normal_transform)
        batch_size, num_workers: DataLoader 설정
        device: torch.device (None 이면 자동 선택)
        pretrained_plain: True 이면 ImageNet pretrained ResNet18 사용

    Returns:
        features_cutpaste: (N, D) CutPaste feature
        features_plain:    (N, D) Plain ResNet feature
        gaussian_cutpaste: {"mu", "cov"}
        gaussian_plain:    {"mu", "cov"}
        plain_resnet:      Plain ResNet backbone (fc=Identity)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3개 카테고리 정상 이미지 전체 경로
    all_normal_paths = list(
        chain.from_iterable(normal_image_paths[cat] for cat in categories)
    )

    dataset = NormalImageDataset(all_normal_paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # CutPaste feature
    features_cutpaste = extract_features(feature_extractor, loader, device)
    gaussian_cutpaste = fit_gaussian(features_cutpaste)

    # Plain ResNet feature
    plain_resnet = models.resnet18(pretrained=pretrained_plain)
    plain_resnet.fc = nn.Identity()
    features_plain = extract_features(plain_resnet, loader, device)
    gaussian_plain = fit_gaussian(features_plain)

    return (
        features_cutpaste,
        features_plain,
        gaussian_cutpaste,
        gaussian_plain,
        plain_resnet,
    )


__all__ = [
    "NormalImageDataset",
    "extract_features",
    "fit_gaussian",
    "compute_image_gaussians",
]




