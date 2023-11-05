import torch
import numpy as np

from i2iTranslation.constant import *
from i2iTranslation.utils.util import read_image, unnormalize, clip_img, tensor2img, upsample_image

class PostProcess:
    def __init__(
            self,
            args,
            is_apply_tissue_mask: bool=True,
            **kwargs
    ):
        super().__init__()
        self.patch_size = args['train.data.patch_size']
        self.upsample = args['train.data.downsample']
        self.n_channels = args['train.data.output_nc']
        self.patch_norm = args['train.params.patch_norm']
        self.patch_clip = args['train.params.patch_clip']
        self.is_apply_tissue_mask = is_apply_tissue_mask
        self.core_tissue_masks_path = args['src_core_tissue_masks_path']

    def postprocess(
            self,
            src_image_path: str,
            dst_image_path: str,
            dst_patches_fake: torch.Tensor,
            src_patch_coords: np.array,
    ):
        self.src_image_path = src_image_path
        self.dst_image_path = dst_image_path
        self.src_patch_coords = src_patch_coords

        # Patch-level preprocessing
        if self.patch_norm:
            dst_patches_fake = unnormalize(dst_patches_fake, PATCH_NORMALIZATION_MEAN, PATCH_NORMALIZATION_STD)
        if self.patch_clip:
            dst_patches_fake = clip_img(dst_patches_fake)         # clip patch values
        dst_patches_fake = tensor2img(dst_patches_fake)           # convert tensor to PIL

        # Generate dst image
        self.dst_img_stitch = self._stitch_patches(dst_patches_fake)     # stitch patches to create complete image

        # Read src images
        self.src_image = read_image(self.src_image_path)

        # Post-process images
        self._format_image()                                        # image resize and tissue masking (src, dst)

        return self.dst_img_stitch

    def _stitch_patches(self, dst_patches_fake):
        # define dst image dimensions
        ctr_y = len(np.unique(self.src_patch_coords[:, 0]))
        ctr_x = len(np.unique(self.src_patch_coords[:, 1]))
        image_size = (
            int(ctr_y * self.patch_size),
            int(ctr_x * self.patch_size),
            self.n_channels
        )
        dst_img_stitch = np.zeros(shape=image_size)

        # stitching
        for i in range(self.src_patch_coords.shape[0]):
            y, x = self.src_patch_coords[i]
            dst_img_stitch[y:y + self.patch_size, x:x + self.patch_size, :] = \
                upsample_image(dst_patches_fake[i], upsample=self.upsample)
        return dst_img_stitch

    def _get_tissue_mask(self):
        tissue_mask_path = os.path.join(self.core_tissue_masks_path, os.path.basename(self.src_image_path))
        if os.path.isfile(tissue_mask_path):
            return read_image(tissue_mask_path)
        else:
            print('ERROR: tissue mask missing')
            exit()

    def _format_image(self):
        # cropping dst image stitched to match src size
        h, w, _ = self.src_image.shape
        self.dst_img_stitch = self.dst_img_stitch[:h, :w, :]

        # tissue region masking
        if self.is_apply_tissue_mask:
            tissue_mask = self._get_tissue_mask()
            bg_pixels = self.src_image[tissue_mask == 0]
            mean_bg_pixel_value = np.mean(bg_pixels, axis=0)
            self.dst_img_stitch[tissue_mask == 0] = mean_bg_pixel_value


