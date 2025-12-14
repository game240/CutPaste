import os
from glob import glob
from typing import Dict, List, Sequence, Union, Tuple, Optional

from PIL import Image
import torch
from torchvision import transforms


# 기본 이미지 크기 (self-supervised / Gaussian 학습 전 과정에서 동일하게 사용)
IMAGE_SIZE: Tuple[int, int] = (256, 256)

# 기본 카테고리 목록
DEFAULT_CATEGORIES: Tuple[str, ...] = ("bottle", "hazelnut", "tile")


def get_normal_transform(image_size: Tuple[int, int] = IMAGE_SIZE) -> transforms.Compose:
    """MVTec 정상 이미지를 256×256×3 텐서로 만드는 공통 변환을 생성합니다."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _normalize_categories(
    categories: Union[str, Sequence[str], None]
) -> List[str]:
    """카테고리 입력을 일관된 리스트 형태로 정규화합니다.

    - None        → DEFAULT_CATEGORIES
    - "bottle"    → ["bottle"]
    - ["bottle", "tile"] 그대로 유지
    """
    if categories is None:
        return list(DEFAULT_CATEGORIES)
    if isinstance(categories, str):
        return [categories]
    return list(categories)


def collect_normal_image_paths(
    root: str,
    categories: Union[str, Sequence[str], None] = None,
) -> Dict[str, List[str]]:
    """각 카테고리별 정상(train/good) 이미지 경로를 수집합니다.

    Args:
        root: MVTec 데이터 루트 (예: "/mnt/.../mvtec")
        categories:
            - None           → DEFAULT_CATEGORIES 사용
            - "bottle"       → 해당 단일 카테고리만
            - ["bottle",...] → 전달된 리스트 그대로 사용
    """
    cats = _normalize_categories(categories)

    paths_per_cat: Dict[str, List[str]] = {}
    for cat in cats:
        pattern = os.path.join(root, cat, "train", "good", "*")
        paths = sorted(glob(pattern))
        if len(paths) == 0:
            print(f"[경고] 카테고리 '{cat}' 에서 정상 이미지가 발견되지 않았습니다: {pattern}")
        paths_per_cat[cat] = paths
    return paths_per_cat


def load_image_as_tensor(
    path: str,
    transform: Optional[transforms.Compose] = None,
) -> torch.Tensor:
    """단일 이미지를 로드해서 (3×H×W) 텐서로 변환합니다.

    Args:
        path: 이미지 파일 경로
        transform: 사용할 torchvision 변환.
            - None 이면 get_normal_transform(IMAGE_SIZE)를 기본 사용
    """
    img = Image.open(path).convert("RGB")
    if transform is None:
        transform = get_normal_transform(IMAGE_SIZE)
    return transform(img)


__all__ = [
    "IMAGE_SIZE",
    "DEFAULT_CATEGORIES",
    "get_normal_transform",
    "collect_normal_image_paths",
    "load_image_as_tensor",
]


