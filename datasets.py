import os
from typing import List, Tuple

from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset

class BirdSoundDataset(Dataset):

    def __init__(self, dataroot: str, resolution: Tuple[int, int]):
        super().__init__()
        self.dataroot: str = dataroot
        self.resolution: Tuple[int, int] = resolution
        self.image_folder: str = f'{self.dataroot}/images'
        self.mask_folder: str = f'{self.dataroot}/masks'
        self.image_filenames: List[str] = sorted(os.listdir(self.image_folder))
        self.mask_filenames: List[str] = sorted(os.listdir(self.mask_folder))
        assert len(self.image_filenames) == len(self.mask_filenames)

        self.__transformer = T.Compose([
            T.ToTensor(),
            T.Grayscale(num_output_channels=1),
            T.Resize(size=self.resolution),
        ])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_filename: str = self.image_filenames[idx]
        mask_filename: str = self.mask_filenames[idx]
        assert image_filename == mask_filename
        image_path: str = f'{self.image_folder}/{image_filename}'
        mask_path: str = f'{self.mask_folder}/{mask_filename}'
        image_tensor: torch.Tensor = self.__transformer(Image.open(image_path))
        mask_tensor: torch.Tensor = self.__transformer(Image.open(mask_path))
        mask_tensor: torch.Tensor = (mask_tensor != 0).to(dtype=torch.int8)
        return image_tensor, mask_tensor
    
    def __len__(self) -> int:
        return len(self.image_filenames)


# TEST:
if __name__ == '__main__':
    self = BirdSoundDataset(dataroot='data/train', resolution=(128, 512))
    image, mask = self[0]
    print(image.shape)
    print(mask.shape)

