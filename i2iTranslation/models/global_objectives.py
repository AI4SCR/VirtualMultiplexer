import torch.nn as nn
import mlflow
from i2iTranslation.utils.util import robust_mlflow
from i2iTranslation.models.vgg import *
from i2iTranslation.constant import *


def convert_patches_to_image(args, src_patches, src_img_size, src_patch_coords, dst_patches, dst_img_size, dst_patch_coords, netG, device):
    patch_norm = args['train.params.patch_norm']
    patch_clip = args['train.params.patch_clip']
    patch_size = args['train.data.patch_size']
    downsample = args['train.data.downsample']
    patch_batch_size = args['train.params.patch_batch_size']

    # Patch-level post-processing
    def postprocess_image(image: torch.Tensor) -> torch.Tensor:
        if patch_norm:
            image = image * PATCH_NORMALIZATION_STD + PATCH_NORMALIZATION_MEAN      # un-normalize CycleGAN normalizer
        if patch_clip:
            image = torch.clip(image, 0.0, 1.0)
        return image

    # initialize reconstructed images. img_sizes and co-ordinates for both src and dst are already downsampled
    src_img = torch.zeros(size=src_img_size, device=device)
    src_mask = torch.zeros_like(src_img)[0, :, :]
    dst_img = torch.zeros(size=dst_img_size, device=device)
    dst_mask = torch.zeros_like(dst_img)[0, :, :]
    dst_img_fake = torch.zeros(size=src_img_size, device=device)
    patch_size = int(patch_size // downsample)

    # reconstruct src real images
    for k, (y, x) in enumerate(src_patch_coords):
        src_img[:, y: y + patch_size, x: x + patch_size] = src_patches[k]
        src_mask[y: y + patch_size, x: x + patch_size] = 1
    src_img = postprocess_image(src_img)

    # reconstruct dst real images
    for k, (y, x) in enumerate(dst_patch_coords):
        dst_img[:, y: y + patch_size, x: x + patch_size] = dst_patches[k]
        dst_mask[y: y + patch_size, x: x + patch_size] = 1
    dst_img = postprocess_image(dst_img)

    # reconstruct dst fake images
    for j in range(0, len(src_patches), patch_batch_size):
        src_patches_ = src_patches[j: j + patch_batch_size]
        src_patches_ = src_patches_.to(device)
        dst_patches_fake = netG(src_patches_)
        for k, (y, x) in enumerate(src_patch_coords[j: j + patch_batch_size]):
            dst_img_fake[:, y: y + patch_size, x: x + patch_size] = dst_patches_fake[k]
    dst_img_fake_ = postprocess_image(dst_img_fake)

    return src_img, src_mask, dst_img, dst_mask, dst_img_fake_

def gram_matrix(x, normalize=True):
    '''
    Generate gram matrices of the representations of content and style images.
    '''
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if normalize:
        gram /= ch * h * w
    return gram


class SCLossCriterion(nn.Module):
    def __init__(self, args, device):
        super(SCLossCriterion, self).__init__()

        self.device = device
        self._prepare_model(args)

        content_layer_names = ''
        for x in self.content_layer_names:
            content_layer_names = content_layer_names + x + ','
        style_layer_names = ''
        for x in self.style_layer_names:
            style_layer_names = style_layer_names + x + ','

        robust_mlflow(mlflow.log_param, 'content_layer_names', content_layer_names)
        robust_mlflow(mlflow.log_param, 'style_layer_names', style_layer_names)

    def _prepare_model(self, args):
        # we are not tuning model weights -> requires_grad=False
        image_model_name = args['train.params.image_model_name']
        if image_model_name == 'vgg16':
            model = Vgg16(requires_grad=False)
        elif image_model_name == 'vgg19':
            model = Vgg19(requires_grad=False)
        elif image_model_name == 'vgg16experimental':
            model = Vgg16Experimental(requires_grad=False)
        else:
            raise ValueError(f'{image_model_name} not supported.')

        self.content_layer_names = model.content_layer_names
        self.style_layer_names = model.style_layer_names
        self.model = model.to(self.device).eval()

    def _content_loss(self, real_content, fake_content):
        real_content = real_content.detach()
        return nn.MSELoss(reduction='mean')(real_content, fake_content)

    def _style_loss(self, real_style, fake_style, weighted=True):
        real_style = real_style.detach()     # we dont need the gradient of the target
        size = real_style.size()

        if not weighted:
            weights = torch.ones(size=real_style.shape[0])
        else:
            # https://arxiv.org/pdf/2104.10064.pdf
            Nl = size[1] * size[2]  # C x C = C^2
            real_style_norm = torch.linalg.norm(real_style, dim=(1, 2))
            fake_style_norm = torch.linalg.norm(fake_style, dim=(1, 2))
            normalize_term = torch.square(real_style_norm) + torch.square(fake_style_norm)
            weights = Nl / normalize_term

        se = (real_style.view(size[0], -1) - fake_style.view(size[0], -1)) ** 2
        return (se.mean(dim=1) * weights).mean()

    def forward(self, content_img, style_img, fake_img):
        content_img_feature_maps = self.model(content_img)
        style_img_feature_maps = self.model(style_img)
        fake_img_feature_maps = self.model(fake_img)

        real_content_representation = [x for cnt, x in enumerate(content_img_feature_maps) if cnt in self.model.content_feature_maps_indices]
        real_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_feature_maps) if cnt in self.model.style_feature_maps_indices]

        fake_content_representation = [x for cnt, x in enumerate(fake_img_feature_maps) if cnt in self.model.content_feature_maps_indices]
        fake_style_representation = [gram_matrix(x) for cnt, x in enumerate(fake_img_feature_maps) if cnt in self.model.style_feature_maps_indices]

        # content loss
        content_loss = 0
        for i, layer in enumerate(self.content_layer_names):
            content_loss += self._content_loss(real_content_representation[i], fake_content_representation[i])

        # style loss
        style_loss = 0
        for i, layer in enumerate(self.style_layer_names):
            style_loss += self._style_loss(real_style_representation[i], fake_style_representation[i])

        return content_loss, style_loss












