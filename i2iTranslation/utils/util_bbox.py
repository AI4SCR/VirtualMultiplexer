from typing import List
import os
import torch
import numpy as np

def load_bboxes(patch_paths: List[str], downsample: int, bbox_info_path: str) -> List[np.ndarray]:
    bboxes = []
    for path in patch_paths:
        basename = os.path.basename(path)[:-4]  # remove image extension .png

        # y0, x0, y1, x1, roiclass (0: pos roi, 1: neg roi)
        bbox = np.load(os.path.join(bbox_info_path, basename + '.npz'))
        bbox = bbox['bbox']

        if downsample != 1:
            if bbox.size != 0:
                if bbox.ndim == 1:
                    bbox = np.reshape(bbox, newshape=(1, -1))

                bbox[:, :-1] = bbox[:, :-1].astype(float) / float(downsample)
                bbox = bbox.astype(int)

        bboxes.append(bbox)
    return bboxes

def draw_conditions(bboxes, dim, num_classes=2):
    condition = torch.zeros((1, num_classes, dim, dim))

    for i, bbox in enumerate(bboxes):
        y0, x0, y1, x1, cls = map(int, bbox)
        condition[0, cls, y0:y1, x0:x1] = 1

    return condition

def bbox_data_generator(patches: np.ndarray, bboxes: List[np.ndarray], nb_rois: int = 8):
    patch_dim = patches.shape[2]

    # sample a set of nuclei bboxes from each bbox set (corresponding to a patch)
    bbox_batch = []         # [#patches[5]]
    idx_patches_no_bbox = []
    for i, bbox in enumerate(bboxes):
        # check if the patch has nuclei bboxes
        if bbox.size == 0:
            idx_patches_no_bbox.append(i)
            continue

        # if only pos nuclei (pos nuclei label: 0)
        if sum(bbox[:, -1]) == 0:
            idx = np.random.choice(len(bbox), 5 * (nb_rois // 4), replace=True)     # oversample POS nuclei

        # if only neg nuclei (neg nuclei label: 1)
        elif sum(bbox[:, -1]) == len(bbox):
            idx = np.random.choice(len(bbox), 3 * (nb_rois // 4), replace=True)     # undersample NEG nuclei

        # if both pos and neg nuclei
        elif sum(bbox[:, -1]) < len(bbox):
            idx_pos = np.where(bbox[:, -1] == 0)[0]
            idx_pos = np.random.choice(idx_pos, nb_rois, replace=True)              # retain sample POS nuclei
            idx_neg = np.where(bbox[:, -1] == 1)[0]
            idx_neg = np.random.choice(idx_neg, nb_rois // 2, replace=True)         # undersample NEG nuclei
            idx = np.concatenate((idx_pos, idx_neg), axis=None)

        bbox_batch.append(bbox[idx])

    # remove patches without any nuclei bboxes
    if len(idx_patches_no_bbox) != 0:
        keep_idx = np.delete(np.arange(len(patches)), np.array(idx_patches_no_bbox))
        patches = patches[keep_idx]

    # convert sampled bboxes per patch into pos-neg masks - [batch_size, 2, H, W]   ## 2 classes (pos, neg)
    condition_batch = []
    for i in range(len(bbox_batch)):
        condition = draw_conditions(bbox_batch[i], patch_dim)
        condition_batch.append(condition)

    condition_batch = torch.cat(condition_batch, axis=0)

    return patches, condition_batch, bbox_batch


