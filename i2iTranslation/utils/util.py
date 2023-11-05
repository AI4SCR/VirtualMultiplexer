"""This module contains simple helper functions """
from __future__ import print_function
from typing import Dict, Any, List
import os
import itertools
import tempfile
import importlib
import time
from pathlib import Path
import yaml
import collections
import mlflow
from mlflow.tracking import MlflowClient
from http.client import RemoteDisconnected

import numpy as np
from scipy import signal
import pandas as pd
from sklearn.metrics import confusion_matrix
import cv2
from matplotlib import pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms



def start_train_experiment(args):
    mlflow.set_experiment(args['mlflow_experiment'])
    mlflow.start_run()

    # Get active experiment_id and run_id
    if mlflow.active_run() is not None:
        experiment_id = robust_mlflow(mlflow.active_run).info.experiment_id
        active_run = robust_mlflow(mlflow.active_run)
        run_id = active_run.info.run_id
    else:
        experiment_id, run_id = '00', '00'

    return experiment_id, run_id

def start_test_experiment(args):
    mlflow.set_experiment(args['mlflow_experiment'])
    mlflow.start_run()

    # Log parameters
    mlflow.log_params(args)

def read_mlflow(args={}, mlflow_run_id='', excluded_keys=[]):
    client = MlflowClient()
    run = client.get_run(mlflow_run_id)
    run_params = run.data.params

    for key, val in run_params.items():
        if key not in excluded_keys:
            f = convert_type(val)
            args[key] = f(val)
    return args

def robust_mlflow(f, *args, max_tries=8, delay=1, backoff=2, **kwargs):
    while max_tries > 1:
        try:
            return f(*args, **kwargs)
        except RemoteDisconnected:
            print(f"MLFLOW remote disconnected. Trying again in {delay}s")
            time.sleep(delay)
            max_tries -= 1
            delay *= backoff
    return f(*args, **kwargs)

def get_split(splits_path: str, split_key: str):
    '''
    Args:
        splits_dir (string): Path to the csv file of all splits.
        split_key (string): Split key.
    '''
    all_splits = pd.read_csv(splits_path)
    split = all_splits[split_key]
    split = split.dropna().reset_index(drop=True)
    return split

def set_seed(device, seed=0):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def random_seed():
    return np.random.seed(int(1000 * time.time()) % 2 ** 32)

def rescale_tensor(img, scale_factor):
    if img.ndim == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:
        img = img.unsqueeze(0)

    return F.interpolate(img, scale_factor=(scale_factor, scale_factor), mode='bilinear', align_corners=True).squeeze()

def downsample_image(img, downsample):
    new_size = (
        int(img.shape[1] // downsample),
        int(img.shape[0] // downsample),
    )
    img = cv2.resize(
        img,
        new_size,
        interpolation=cv2.INTER_LINEAR,
    )
    return img

def upsample_image(img, upsample):
    new_size = (
        int(img.shape[0] * upsample),
        int(img.shape[1] * upsample),
    )
    img = cv2.resize(
        img,
        new_size,
        interpolation=cv2.INTER_LINEAR,
    )
    return img

def delete_tensor_gpu(tensor_dict: Dict):
    for k, v in tensor_dict.items():
        del v

def convert_type(val):
    if val.isdigit():
        return int
    elif val.replace('.', '').isdigit() or '1e-' in val:
        return float
    elif val.lower() in ['true', 'false']:
        return eval
    else:
        return str

def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def read_image(path: str):
    return np.asarray(Image.open(path))

def save_image(image: np.ndarray, path: str):
    Image.fromarray(image).save(path)

def save_tissue_mask(
        mask: np.ndarray,
        path: str,
        color_palette: list = []
):
    if len(color_palette) == 0:
        color_palette = [0, 0, 0,               # non-tissue is black
                         255, 255, 255]         # tissue is white

    mask = Image.fromarray(mask)
    mask.putpalette(color_palette)
    mask.convert('RGB').save(path)

def save_roi_mask(mask, save_path) :
    color_palette = [0, 0, 0,          # background
                     255, 0, 0,        # RoI
                     0, 0, 255]        # Non-RoI

    hm = Image.fromarray(mask.astype('uint8'), 'P')
    hm.putpalette(color_palette)
    hm.save(save_path)


def plot(image: np.ndarray, cmap: str = ''):
    if cmap != '':
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    plt.show()



def unnormalize(patches: torch.Tensor, norm_mean: float, norm_std: float):
    patches = [x * norm_std + norm_mean for x in patches]
    return patches

def clip_img(patches: torch.Tensor):
    patches = [torch.clip(x, 0.0, 1.0) for x in patches]
    return patches

def tensor2img(patches: torch.Tensor) -> List[np.ndarray]:
    transform = transforms.Compose([transforms.ToPILImage()])
    patches = [np.array(transform(x.squeeze())) for x in patches]
    return patches

def gkern(kernlen=1, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.windows.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def get_kernel(padding=1, gaussian_std=3, mode="constant"):
    kernel_size = padding * 2 + 1
    if mode == "constant":
        kernel = torch.ones(kernel_size, kernel_size)
        kernel = kernel / (kernel_size * kernel_size)

    elif mode == "gaussian":
        kernel = gkern(kernel_size, std=gaussian_std)
        kernel = kernel / kernel.sum()
        kernel = torch.from_numpy(kernel.astype(np.float32))

    else:
        raise NotImplementedError

    return kernel

def save_roi_mask(mask, save_path) :
    color_palette = [0, 0, 0,          # background
                     255, 0, 0,        # RoI
                     0, 0, 255]        # Non-RoI

    hm = Image.fromarray(mask.astype('uint8'), 'P')
    hm.putpalette(color_palette)
    hm.save(save_path)

def log_confusion_matrix(prediction, ground_truth, classes, name):
    cm = confusion_matrix(
        y_true=ground_truth, y_pred=prediction, labels=np.arange(len(classes))
    )
    fig = plot_confusion_matrix(cm, classes, figname=None, normalize=False)
    save_fig_to_mlflow(fig, "confusion_plots", name)

def plot_confusion_matrix(
    cm, classes, figname=None, normalize=False, title=None, cmap=plt.cm.Oranges
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "%.2f" % cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
            )
        else:
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16,
            )
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    ax.imshow(cm, interpolation="nearest", cmap=cmap)
    if title is not None:
        ax.set_title(title)
    return fig

