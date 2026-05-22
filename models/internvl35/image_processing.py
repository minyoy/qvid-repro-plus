from itertools import product
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def _closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    best_ratio = (1, 1)
    best_ratio_diff = float("inf")
    area = width * height

    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> List[Image.Image]:
    width, height = image.size
    target_ratios = sorted(
        {
            (cols, rows)
            for tiles in range(min_num, max_num + 1)
            for cols, rows in product(range(1, tiles + 1), repeat=2)
            if min_num <= cols * rows <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    cols, rows = _closest_aspect_ratio(
        aspect_ratio=width / height,
        target_ratios=target_ratios,
        width=width,
        height=height,
        image_size=image_size,
    )
    resized = image.convert("RGB").resize(
        (image_size * cols, image_size * rows),
        BICUBIC,
    )
    processed = []
    for tile_idx in range(cols * rows):
        left = (tile_idx % cols) * image_size
        top = (tile_idx // cols) * image_size
        processed.append(
            resized.crop((left, top, left + image_size, top + image_size))
        )

    if use_thumbnail and len(processed) != 1:
        processed.append(
            image.convert("RGB").resize(
                (image_size, image_size),
                BICUBIC,
            )
        )

    return processed


def image_to_pixel_values(
    image: Image.Image,
    input_size: int = 448,
    max_num_tiles: int = 12,
) -> torch.Tensor:
    tiles = dynamic_preprocess(
        image=image,
        image_size=input_size,
        max_num=max_num_tiles,
        use_thumbnail=True,
    )
    tensors = []
    for tile in tiles:
        array = np.asarray(tile, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append((tensor - IMAGENET_MEAN) / IMAGENET_STD)
    return torch.stack(tensors)
