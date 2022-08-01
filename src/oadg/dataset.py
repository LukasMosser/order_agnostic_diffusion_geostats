import os
import os.path
import warnings
from typing import Any, Callable, Optional, Tuple
from urllib.error import URLError

import numpy as np
from PIL import Image
import imageio

from torchvision.datasets.utils import check_integrity, download_url
from torchvision.datasets.vision import VisionDataset


class Channels(VisionDataset):
    mirrors = [
        "https://raw.githubusercontent.com/elaloy/gan_for_gradient_based_inv/master/training/ti/",
    ]

    resources = [
        ("ti.png", "2056cd0456867554c4f9d28b50ce671f"),
    ]

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        download: bool = False,
        patch_size: int = 64,
        stride: int = 16,
        flip_up: bool = True,
        flip_left: bool = True,
        mirror: bool = True
    ) -> None:
        super().__init__(root, transform=transform)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data = self._extract_patches(patch_size, stride, flip_up, flip_left, mirror)

    def _create_dataset(self, patch_size: int, stride: int, flip_up: bool, flip_left: bool, mirror: bool):
        root = os.path.expanduser(self.raw_folder)
        fpath = os.path.join(root, "ti.png")

        img = imageio.imread(fpath)

        patches = self._extract_patches(img, patch_size=patch_size, stride=stride,
                                        flip_up_down=flip_up, flip_left_right=flip_left, mirror=mirror)

        return patches

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}{filename}"
                try:
                    print(f"Downloading {url}")
                    download_url(url, root=self.raw_folder, filename=filename, md5=md5)
                except URLError as error:
                    print(f"Failed to download (trying next):\n{error}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    @staticmethod
    def _extract_patches(img, patch_size=64, stride=16, flip_up_down=False, flip_left_right=False, mirror=False):
        patches = []
        for i in range(0, img.shape[0] - patch_size, stride):
            for j in range(0, img.shape[1] - patch_size, stride):
                patches.append(img[i:i + patch_size, j:j + patch_size])
        patches = np.array(patches)
        all_patches = [patches]

        if flip_up_down:
            all_patches.append(patches[..., ::-1, :])

        if flip_left_right:
            all_patches.append(patches[..., :, ::-1])

        if mirror:
            all_patches.append(patches[..., ::-1, ::-1])

        patches = np.concatenate(all_patches, axis=0)
        return patches
