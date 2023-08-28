# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torchvision.transforms as transforms
from ubteacher.data.transforms.augmentation_impl import (
    GaussianBlur,
)

import torch

class RandomErasingInTiles(transforms.RandomErasing):
    def __init__(self, tiles=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiles = tiles

    def forward(self, img):
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        h_step, w_step = img_h // self.tiles, img_w // self.tiles

        for i in range(self.tiles):
            for j in range(self.tiles):
                y_start = i * h_step
                x_start = j * w_step
                tile_img = img[:, y_start:y_start+h_step, x_start:x_start+w_step]

                if torch.rand(1) < self.p:
                    if isinstance(self.value, (int, float)):
                        value = [float(self.value)]
                    elif isinstance(self.value, str):
                        value = None
                    elif isinstance(self.value, (list, tuple)):
                        value = [float(v) for v in self.value]
                    else:
                        value = self.value

                    if value is not None and not (len(value) in (1, tile_img.shape[-3])):
                        raise ValueError(
                            "If value is a sequence, it should have either a single value or "
                            f"{tile_img.shape[-3]} (number of input channels)"
                        )

                    y, x, h, w, v = self.get_params(tile_img, scale=self.scale, ratio=self.ratio, value=value)
                    img[:, y_start+y:y_start+y+h, x_start+x:x_start+x+w] = v

        return img

def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        augmentation.append(
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomErasing(
                    p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                ),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                ),
                transforms.RandomErasing(
                    p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                ),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(randcrop_transform)
        """
        # RandomErasingInTiles
        tile_randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                RandomErasingInTiles(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random", tiles=4),
                RandomErasingInTiles(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random", tiles=4),
                RandomErasingInTiles(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random", tiles=4),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(tile_randcrop_transform)

        # ErasingInTiles
        tile_randcrop_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                RandomErasingInTiles(p=1, scale=(0.25,0.25), ratio=(0.3, 3.3), value="random", tiles=4),
                transforms.ToPILImage(),
            ]
        )
        augmentation.append(tile_randcrop_transform)
        """

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)