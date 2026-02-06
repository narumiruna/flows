from typing import Literal

from PIL import Image
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageNet

from .settings import settings


def distributed_is_initialized() -> bool:
    return distributed.is_initialized() and distributed.is_available()  # ty:ignore[possibly-missing-attribute]


class ImageNetLoader(DataLoader):
    def __init__(
        self,
        split: Literal["train", "val"] = "train",
        image_size: int = 224,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> None:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if split == "train":
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, interpolation=Image.Resampling.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        elif split == "val":
            transform = transforms.Compose(
                [
                    transforms.Resize(image_size + 32, interpolation=Image.Resampling.BICUBIC),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        dataset = ImageNet(
            settings.dataset_root,
            split=split,
            transform=transform,
        )

        sampler = None
        if split == "train" and distributed_is_initialized():
            sampler = DistributedSampler(dataset)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
        )
