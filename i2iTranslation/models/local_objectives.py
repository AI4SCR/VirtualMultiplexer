import torch
import torch.nn as nn
from torchvision.ops import RoIAlign
import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class DiscriminatorROI(nn.Module):

    def __init__(self, base_filters=16):
        super(DiscriminatorROI, self).__init__()

        self.conv_layers = nn.Sequential(
            *self.conv_block(5, base_filters, normalise=False),
            *self.conv_block(1 * base_filters, 2 * base_filters),
            *self.conv_block(2 * base_filters, 4 * base_filters),
            *self.conv_block(4 * base_filters, 8 * base_filters))

        self.roi_pool = RoIAlign(output_size=(3, 3), spatial_scale=0.0625, sampling_ratio=-1)

        self.classifier = nn.Sequential(
            nn.Conv2d(8 * base_filters, 1, kernel_size=3, padding=0, bias=False))

        self.apply(weights_init_normal)

    def conv_block(self, in_filters, out_filters, normalise=True):
        layers = [
            nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalise:
            layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, inputs, condition, bboxes):
        bbox_batch = bboxes[:, :-1]

        x = torch.cat([inputs, condition], axis=1)
        x = self.conv_layers(x)
        pool = self.roi_pool(x, bbox_batch)
        outputs = self.classifier(pool)

        return outputs.squeeze()


class ClassifierROI(nn.Module):

    def __init__(self, base_filters=16, roi_patch_size=32, num_classes=2):
        super(ClassifierROI, self).__init__()

        self.roi_size = (roi_patch_size, roi_patch_size)
        self.kernel_size = 3
        self.padding = 1

        self.conv_layers = nn.Sequential(
            *self.conv_block(3, base_filters, normalise=False),
            *self.conv_block(1 * base_filters, 2 * base_filters),
            *self.conv_block(2 * base_filters, 4 * base_filters))

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4 * base_filters, num_classes)

        self.apply(weights_init_normal)

    def conv_block(self, in_filters, out_filters, normalise=True):
        layers = [nn.Conv2d(in_filters, out_filters, kernel_size=self.kernel_size, stride=2, padding=self.padding, bias=False)]
        if normalise:
            layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
        layers.append(nn.ReLU())
        return layers

    def forward(self, inputs, bboxes):
        # roi data preparation
        inputs_roi = []
        for bbox in bboxes:
            input_idx, y0, x0, y1, x1, cls = bbox
            roi = inputs[input_idx, :, y0:y1, x0:x1]
            resized_roi = F.interpolate(roi.unsqueeze(0), size=self.roi_size, mode='bilinear', align_corners=False)
            inputs_roi.append(resized_roi.squeeze(0))

        inputs_roi = torch.stack(inputs_roi)
        labels_roi = bboxes[:, -1]

        # forward pass
        x = self.conv_layers(inputs_roi)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        outputs_roi = self.classifier(x)

        return labels_roi.float(), outputs_roi