from typing import Optional, Callable, Any, List, Tuple, Dict
import os
import glob
import random
from tqdm import tqdm
import mlflow
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from i2iTranslation.constant import *
from i2iTranslation.utils.util import get_split, robust_mlflow, read_image, downsample_image, random_seed, rescale_tensor


class BaseDataset(Dataset):
    def __init__(
            self,
            args,
            image_fmt: str='png',
            **kwargs
    ) -> None:
        self.is_train = args['is_train']
        self.src_marker = args['src_marker']
        self.dst_marker = args['dst_marker']
        self.patch_size = args['train.data.patch_size']
        self.patch_norm = args['train.params.patch_norm']
        self.max_src_samples = args['train.data.max_src_samples']
        self.downsample = args['train.data.downsample']
        self.core_images_path = args['core_images_path']
        self.i2i_patches_path = args['i2i_patches_path']
        self.image_fmt = image_fmt
        self.is_i2i_stage2 = True if 'i2i_stage2' in args.keys() else False

        self.downsampled_size = (
            int(self.patch_size // self.downsample),
            int(self.patch_size // self.downsample),
        )
        self.mode = 'train' if self.is_train else 'test'

        print('Source marker: ', self.src_marker)
        print('Destination marker: ', self.dst_marker)

        # source and destination file names
        self.src_split = get_split(splits_path=os.path.join(args['src_data_split_path']), split_key=f'{self.mode}_cores')
        self.dst_split = get_split(splits_path=os.path.join(args['dst_data_split_path']), split_key=f'{self.mode}_cores')

        # define src and dst splits
        self._get_splits()

        # load image and patch paths for src and dst splits
        self._base_load()

    def _get_splits(self):
        self.src_split = self.src_split.values.tolist()
        self.dst_split = self.dst_split.values.tolist()

        # select subset of src cores for training
        if self.is_train and (self.max_src_samples != -1):
            assert self.max_src_samples <= len(self.src_split), "ERROR: max src samples is greater than src split"
            np.random.seed(0)
            idx = np.random.choice(len(self.src_split), size=min(self.max_src_samples, len(self.src_split)), replace=False)
            self.src_split = [self.src_split[x] for x in idx]

        # select common src and dst cores
        src_split = []
        dst_split = []
        for x in self.src_split:
            y = x.replace(self.src_marker, self.dst_marker)
            if y in self.dst_split:
                src_split.append(x)
                dst_split.append(y)
        self.src_split = src_split
        self.dst_split = dst_split

        if self.is_train and (not self.is_i2i_stage2):
            robust_mlflow(mlflow.log_param, f'n_src_ids_{self.mode}', len(self.src_split))
            robust_mlflow(mlflow.log_param, f'n_dst_ids_{self.mode}', len(self.dst_split))

    def _base_load(self):
        # get src and dst image paths
        self.src_image_paths = [f'{self.core_images_path}/{self.src_marker}/{x}.{self.image_fmt}' for x in self.src_split]
        self.dst_image_paths = [f'{self.core_images_path}/{self.dst_marker}/{x}.{self.image_fmt}' for x in self.dst_split]

        if self.is_train:
            # get patch names
            def _get_patch_paths(marker: str, split: List) -> Dict:
                paths_ = dict()
                for core_name in split:
                    paths_[core_name] = glob.glob(f'{self.i2i_patches_path}/{marker}/{core_name}_*.{self.image_fmt}')
                return paths_
            self.src_patch_paths = _get_patch_paths(self.src_marker, self.src_split)
            self.dst_patch_paths = _get_patch_paths(self.dst_marker, self.dst_split)

    def _get_transform(self):
        pass


# Return all patches (corresponding to the same core) from src and dst images.
# It supports both pre-extracted patches (useful for training) and on-the-fly extracted patches (useful for testing).
class ImagePatchDataset(BaseDataset):
    def __init__(self, args, **kwargs) -> None:
        super().__init__(args, **kwargs)
        self.load_in_ram = args['train.data.load_in_ram']

        # sort images to serially export results during testing
        if not self.is_train:
            self.src_image_paths.sort()
            self.dst_image_paths.sort()

        # load image and corresponding patch information
        self._load()

        # data normalization
        self.transform = self._get_transform()

    def _get_shape(self, image_paths):
        image_shape = []
        for x in tqdm(image_paths):
            image = read_image(x)
            h, w, c = image.shape
            pad_h = self.patch_size - h % self.patch_size
            pad_w = self.patch_size - w % self.patch_size
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)
            h, w, c = image.shape
            if self.downsample != 1:
                h = h // self.downsample
                w = w // self.downsample
            image_shape.append((c, h, w))
        return image_shape

    def _load(self):
        if self.is_train:
            # get src and dst image dimensions
            print('extracting image shapes...')

            self.src_image_shape = self._get_shape(self.src_image_paths)
            self.dst_image_shape = self._get_shape(self.dst_image_paths)

            # load data into memory
            if self.load_in_ram:
                print('loading patch information...')
                def _get_data(image_paths, image_patch_paths):
                    patches = dict()
                    patch_coords = dict()
                    patch_paths = dict()
                    for image_path in tqdm(image_paths):
                        image_id = os.path.basename(image_path).split(f'.{self.image_fmt}')[0]
                        patches[image_id], patch_coords[image_id], patch_paths[image_id] = \
                            self._load_patches(image_patch_paths, image_id)
                    return patches, patch_coords, patch_paths
                self.src_patches, self.src_patch_coords, self.src_patch_paths = _get_data(self.src_image_paths, self.src_patch_paths)
                self.dst_patches, self.dst_patch_coords, self.dst_patch_paths = _get_data(self.dst_image_paths, self.dst_patch_paths)

    def _get_transform(self):
        transform_list = []

        transform_list.append(transforms.ToTensor())
        if self.patch_norm:
            transform_list.append(
                transforms.Normalize(
                    (PATCH_NORMALIZATION_MEAN, PATCH_NORMALIZATION_MEAN, PATCH_NORMALIZATION_MEAN),
                    (PATCH_NORMALIZATION_STD, PATCH_NORMALIZATION_STD, PATCH_NORMALIZATION_STD)
                )
            )

        return transforms.Compose(transform_list)

    def _extract_patches(self, path: str) -> Tuple[List[np.ndarray], np.ndarray]:
        # useful during testing
        image = read_image(path)
        h, w, _ = image.shape
        image = np.pad(image, ((0, self.patch_size), (0, self.patch_size), (0, 0)), mode='constant', constant_values=255)
        patches = list()
        patch_coords = list()

        # loop over the padded image to extract patches
        y = 0
        while y <= h:
            x = 0
            while x <= w:
                patch = image[y:y + self.patch_size, x:x + self.patch_size, :]
                if self.downsample != 1:
                    patch = downsample_image(patch, self.downsample)

                patches.append(patch)
                patch_coords.append([y, x])
                x += self.patch_size
            y += self.patch_size
        del image

        return patches, np.array(patch_coords)

    def _load_patches(
            self,
            marker_patch_paths: [],
            core_name: str
    ) -> Tuple[List[Any], List[List[int]], List[str]]:
        # load pre-extracted patches and corresponding information
        paths = marker_patch_paths[core_name]
        patches = list()
        patch_coords = list()
        patch_paths = list()
        for path_ in paths:
            # load patch
            patch = Image.open(path_).convert('RGB')
            if self.downsample != 1:
                patch = patch.resize(self.downsampled_size, resample=Image.BILINEAR)
            patches.append(patch)

            # load patch co-ordinates
            basename = os.path.basename(path_)
            y = int(basename.rpartition('_y_')[2].rpartition('_x_')[0])
            x = int(basename.rpartition('_x_')[2].rpartition('_patch_')[0])
            if self.downsample != 1:
                y = int(y/self.downsample)
                x = int(x/self.downsample)
            patch_coords.append([y, x])

            # load patch paths
            patch_paths.append(os.path.basename(path_))

        return patches, patch_coords, patch_paths

    def _get_patches(
            self,
            src_image_path: str,
            dst_image_path: str,
            is_balance: bool=True
    ) -> Tuple[List[Any], np.ndarray, List[str], List[Any], np.ndarray, List[str]]:
        src_image_id = os.path.basename(src_image_path).split(f'.{self.image_fmt}')[0]
        dst_image_id = os.path.basename(dst_image_path).split(f'.{self.image_fmt}')[0]

        # load patches and corresponding information
        if self.load_in_ram:
            src_patches = self.src_patches[src_image_id]
            src_patch_coords = self.src_patch_coords[src_image_id]
            src_patch_paths = self.src_patch_paths[src_image_id]
            dst_patches = self.dst_patches[dst_image_id]
            dst_patch_coords = self.dst_patch_coords[dst_image_id]
            dst_patch_paths = self.dst_patch_paths[dst_image_id]
        else:
            src_patches, src_patch_coords, src_patch_paths = self._load_patches(self.src_patch_paths, src_image_id)
            dst_patches, dst_patch_coords, dst_patch_paths = self._load_patches(self.dst_patch_paths, dst_image_id)

        # balance the number of src and dst patches
        if is_balance:
            n_patches = min(len(src_patches), len(dst_patches))

            # balance src
            np.random.seed(random_seed())
            idx = np.random.choice(len(src_patches), size=n_patches, replace=False)
            src_patches = [src_patches[x] for x in idx]
            src_patch_coords = [src_patch_coords[x] for x in idx]
            src_patch_paths = [src_patch_paths[x] for x in idx]

            # balance dst
            np.random.seed(random_seed())
            idx = np.random.choice(len(dst_patches), size=n_patches, replace=False)
            dst_patches = [dst_patches[x] for x in idx]
            dst_patch_coords = [dst_patch_coords[x] for x in idx]
            dst_patch_paths = [dst_patch_paths[x] for x in idx]

        else:
            # balance dst according to src
            n_patches_src = len(src_patches)
            n_patches_dst = len(dst_patches)

            np.random.seed(random_seed())
            if n_patches_src < n_patches_dst:
                idx = np.random.choice(n_patches_dst, size=n_patches_src, replace=False)
            elif n_patches_src > n_patches_dst:
                idx = np.random.choice(n_patches_dst, size=n_patches_src, replace=True)
            else:
                idx = np.random.choice(n_patches_dst, size=n_patches_src, replace=False)

            dst_patches = [dst_patches[x] for x in idx]
            dst_patch_coords = [dst_patch_coords[x] for x in idx]
            dst_patch_paths = [dst_patch_paths[x] for x in idx]

        return src_patches, np.array(src_patch_coords), src_patch_paths, dst_patches, np.array(dst_patch_coords), dst_patch_paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, Tuple, Tuple, List[str], List[str]]:
        # source and destination image
        src_image_path = self.src_image_paths[index]
        dst_image_path = self.dst_image_paths[index]

        if self.is_train:
            # load pre-extracted patches
            src_patches, src_patch_coords, src_patch_paths, \
                dst_patches, dst_patch_coords, dst_patch_paths = \
                    self._get_patches(src_image_path, dst_image_path)

            # normalize patches
            src_patches = [self.transform(x) for x in src_patches]
            src_patches = torch.stack(src_patches)
            dst_patches = [self.transform(x) for x in dst_patches]
            dst_patches = torch.stack(dst_patches)

            return src_patches, dst_patches, src_patch_coords, dst_patch_coords, \
                   self.src_image_shape[index], self.dst_image_shape[index], src_patch_paths, dst_patch_paths
        else:
            # extract patches from source
            src_patches, src_patch_coords = self._extract_patches(src_image_path)

            # normalize patches
            src_patches = [self.transform(x) for x in src_patches]
            src_patches = torch.stack(src_patches)

            return src_patches, src_patch_coords, src_image_path, dst_image_path

    def __len__(self) -> int:
        return len(self.src_image_paths)


class TileDataset(Dataset):
    def __init__(self, args):
        self.n_tiles = 8
        self.tile_size = 224
        self.tile_threshold = 0.2
        self.tile_area = self.tile_size * self.tile_size
        self.downsample = args['train.params.image_downsample']
        self._get_transform()

    def _get_transform(self, tile_mean=(0.485, 0.456, 0.406), tile_std=(0.229, 0.224, 0.225)):
        self.transform = transforms.Compose([
            transforms.Normalize(tile_mean, tile_std)
        ])

    def _extract_tile_coordinates(self, image: torch.Tensor, mask: torch.Tensor):
        _, h, w = image.shape
        coordinates = []
        ctr = 0
        while ctr < self.n_tiles:
            y = random.randrange(0, h - self.tile_size)
            x = random.randrange(0, w - self.tile_size)
            tile_mask = mask[y: y + self.tile_size, x: x + self.tile_size]
            if tile_mask.sum().item() / self.tile_area > self.tile_threshold:
                coordinates.append([y, x])
                ctr += 1
        return coordinates

    def __call__(self, src_img: torch.Tensor, src_mask: torch.Tensor, dst_img: torch.Tensor, dst_mask: torch.Tensor, dst_img_fake: torch.Tensor):
        if self.downsample != 1:
            scale_factor = 1/self.downsample
            src_img = rescale_tensor(src_img, scale_factor)
            src_mask = rescale_tensor(src_mask, scale_factor)
            dst_img = rescale_tensor(dst_img, scale_factor)
            dst_mask = rescale_tensor(dst_mask, scale_factor)
            dst_img_fake = rescale_tensor(dst_img_fake, scale_factor)

        self.src_tiles = []
        self.dst_tiles = []
        self.dst_fake_tiles = []

        # src and dst_fake tiles need to have the same content
        coordinates = self._extract_tile_coordinates(src_img, src_mask)
        for (y, x) in coordinates:
            self.src_tiles.append(src_img[:, y: y + self.tile_size, x: x + self.tile_size])
            self.dst_fake_tiles.append(dst_img_fake[:, y: y + self.tile_size, x: x + self.tile_size])

        coordinates = self._extract_tile_coordinates(dst_img, dst_mask)
        for (y, x) in coordinates:
            self.dst_tiles.append(dst_img[:, y: y + self.tile_size, x: x + self.tile_size])

        # transform the tiles (to be fed into vgg)
        self.src_tiles = [self.transform(x) for x in self.src_tiles]
        self.dst_tiles = [self.transform(x) for x in self.dst_tiles]
        self.dst_fake_tiles = [self.transform(x) for x in self.dst_fake_tiles]

        return torch.stack(self.src_tiles), torch.stack(self.dst_tiles), torch.stack(self.dst_fake_tiles)


class MetricDataset:
    def __init__(
        self,
        images_real: List[np.ndarray],
        images_fake: List[np.ndarray],
        tissue_masks_real: List[np.ndarray],
        tissue_masks_fake: List[np.ndarray],
        patch_size: int = 512,
        **kwargs
    ):
        super(MetricDataset, self).__init__()

        assert len(images_real) == len(images_fake), 'ERROR: different lengths of real and fake images'
        self.images_real = images_real
        self.images_fake = images_fake
        self.tissue_masks_real = tissue_masks_real
        self.tissue_masks_fake = tissue_masks_fake

        # compute at low dim to save disk space
        self.patch_size = patch_size
        self.stride = int(self.patch_size / 1)
        self.patch_threshold = int(0.5 * patch_size * patch_size)

    def _get_patches(self, image: np.ndarray, tissue_mask: np.ndarray) -> List[np.ndarray]:
        patches = []
        h, w, _ = image.shape
        image_ = np.pad(image, ((0, self.patch_size), (0, self.patch_size), (0, 0)), mode='constant', constant_values=255)
        tissue_mask_ = np.pad(tissue_mask, ((0, self.patch_size), (0, self.patch_size)), mode='constant', constant_values=0)

        y = 0
        while y <= h:
            x = 0
            while x <= w:
                patch = image_[y:y + self.patch_size, x:x + self.patch_size, :]
                tissue_patch = tissue_mask_[y:y + self.patch_size, x:x + self.patch_size]

                # ignore a patch not overlapping with the tissue area
                if np.sum(tissue_patch)/255.0 >= self.patch_threshold:
                    patches.append(patch)

                x += self.stride
            y += self.stride

        return patches

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_real = self.images_real[index]
        image_fake = self.images_fake[index]
        tissue_mask_real = self.tissue_masks_real[index]
        tissue_mask_fake = self.tissue_masks_fake[index]

        h, w, _ = image_real.shape
        h_, w_, _ = image_fake.shape
        H, W = min(h, h_), min(w, w_)

        # center crop equal sized images, as real and fake images can be of different sizes
        image_real = image_real[int(h/2) - int(H/2) : int(h/2) + int(H/2),
                                int(w/2) - int(W/2) : int(w/2) + int(W/2)]
        image_fake = image_fake[int(h_/2) - int(H/2) : int(h_/2) + int(H/2),
                                int(w_/2) - int(W/2) : int(w_/2) + int(W/2)]

        tissue_mask_real = tissue_mask_real[int(h / 2) - int(H / 2): int(h / 2) + int(H / 2),
                                            int(w / 2) - int(W / 2): int(w / 2) + int(W / 2)]
        tissue_mask_fake = tissue_mask_fake[int(h_ / 2) - int(H / 2): int(h_ / 2) + int(H / 2),
                                            int(w_ / 2) - int(W / 2): int(w_ / 2) + int(W / 2)]

        # get patches
        patches_real = self._get_patches(image_real, tissue_mask_real)
        patches_fake = self._get_patches(image_fake, tissue_mask_fake)

        patches_real = [torch.tensor(x, dtype=torch.uint8) for x in patches_real]
        patches_fake = [torch.tensor(x, dtype=torch.uint8) for x in patches_fake]

        return torch.stack(patches_real).permute(0, 3, 1, 2), \
               torch.stack(patches_fake).permute(0, 3, 1, 2)

    def __len__(self) -> int:
        return len(self.images_real)


def collate_images_patches_batch(batch):
    return batch[0]

def collate_images_batch(batch):
    src_images, dst_images, src_image_paths, dst_image_paths = [], [], [], []

    for (_src_image, _dst_image, _src_image_path, _dst_image_path) in batch:
        if _dst_image is not None:
            src_images.append(_src_image)
            dst_images.append(_dst_image)
            src_image_paths.append(_src_image_path)
            dst_image_paths.append(_dst_image_path)

    if len(src_images) > 0:
        src_images = torch.stack(src_images)
        dst_images = torch.stack(dst_images)
        return src_images, dst_images, src_image_paths, dst_image_paths
    else:
        return None, None, None, None

def prepare_image_patch_dataloader(
        args,
        dataset: Dataset,
        shuffle: bool = False,
        sampler: Optional = None,
        collate_fn: Optional[Callable] = collate_images_patches_batch,
        **kwargs
) -> DataLoader:
    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        batch_size=args['train.params.image_batch_size'],
        num_workers=args['train.params.num_workers']
    )
    return dataloader