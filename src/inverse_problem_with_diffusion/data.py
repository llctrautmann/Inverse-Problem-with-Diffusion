from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageFolderDataset(Dataset):
    """
    All PNGs in a folder as tensors scaled to [-1, 1] (what DDPM noising expects).
    Images are resized to image_size if they differ. To train on your own data,
    put the images in one folder, or replace this class with your own Dataset that
    returns a (C, H, W) tensor in [-1, 1].
    """

    def __init__(self, data_dir, image_size: int = 64, in_channels: int = 1):
        self.paths = sorted(Path(data_dir).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"No .png files found in {Path(data_dir).resolve()}")
        self.mode = "L" if in_channels == 1 else "RGB"
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),  # [0, 1]
                transforms.Normalize([0.5] * in_channels, [0.5] * in_channels),  # [-1, 1]
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert(self.mode))


def make_train_dataloader(config) -> DataLoader:
    return DataLoader(
        ImageFolderDataset(config.data_dir, config.image_size, config.in_channels),
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
