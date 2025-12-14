import random
from typing import Tuple, Callable

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class CutPaste(object):
    """사각형 CutPaste 증강만 사용하는 2-class용 CutPaste 구현."""

    def __init__(self, transform: bool = True):
        """
        Args:
            transform: True 이면 패치에 ColorJitter를 적용.
        """
        if transform:
            self.transform = transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
            )
        else:
            self.transform = None

    @staticmethod
    def crop_and_paste_patch(image, patch_w, patch_h, transform, rotation=False):
        """
        Crop patch from original image and paste it randomly on the same image.

        :image: [PIL] _ original image
        :patch_w: [int] _ width of the patch
        :patch_h: [int] _ height of the patch
        :transform: [binary] _ if True use Color Jitter augmentation
        :rotation: [binary[ _ if True randomly rotates image from (-45, 45) range

        :return: augmented image
        """

        org_w, org_h = image.size
        mask = None

        patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
        patch = image.crop((patch_left, patch_top, patch_right, patch_bottom))
        if transform:
            patch= transform(patch)

        if rotation:
            random_rotate = random.uniform(*rotation)
            patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
            mask = patch.split()[-1]

        # new location
        paste_left, paste_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
        aug_image = image.copy()
        aug_image.paste(patch, (paste_left, paste_top), mask=mask)
        return aug_image

    def cutpaste(self, image, area_ratio = (0.02, 0.15), aspect_ratio = ((0.3, 1) , (1, 3.3))):
        """사각형 패치를 잘라 다른 위치에 붙이는 CutPaste 증강."""
        img_area = image.size[0] * image.size[1]
        patch_area = random.uniform(*area_ratio) * img_area
        patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
        patch_w  = int(np.sqrt(patch_area*patch_aspect))
        patch_h = int(np.sqrt(patch_area/patch_aspect))
        cutpaste = self.crop_and_paste_patch(
            image, patch_w, patch_h, self.transform, rotation=False
        )
        return cutpaste

    def __call__(self, image):
        """원본 이미지와 사각형 CutPaste 증강 이미지를 반환."""
        return image, self.cutpaste(image)


# -------------------------
# 헬퍼 함수 (경로 → 텐서 pair)
# -------------------------

_cutpaste_rect = CutPaste(transform=True)


def make_cutpaste_rect_pair(
    path: str,
    transform: Callable[[Image.Image], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """단일 이미지 경로에서 (정상, 사각형 CutPaste) 텐서 쌍을 생성.

    Args:
        path: 이미지 파일 경로
        transform: PIL 이미지를 3×H×W 텐서로 바꾸는 변환 (예: normal_transform)
    """
    img = Image.open(path).convert("RGB")
    _, aug_img = _cutpaste_rect(img)

    orig_tensor = transform(img)
    aug_tensor = transform(aug_img)
    return orig_tensor, aug_tensor


__all__ = [
    "CutPaste",
    "make_cutpaste_rect_pair",
]

