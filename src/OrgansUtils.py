import os
import random
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap


import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from OrgansUtils import *
import albumentations as A

import json

def load_config(config_path):
    with open(os.path.join('../configs', config_path), 'r') as file:
        return json.load(file)

USED_CLASSES = [0, 1, 2, 3, 6, 7, 8, 9]

def create_colormap_for_labels(label_names: List[str], cmap_name: str = 'tab20') -> LinearSegmentedColormap:
    """
    Creates a colormap with specific colors mapped to label indices from a given colormap.
    """
    cmap = plt.get_cmap(cmap_name)  # Get the colormap
    # Create a color for each label based on its relative position in the label set
    colors = [cmap(i / len(label_names)) for i in range(len(label_names))]
    colors[0] = (0, 0, 0, 1)
    return LinearSegmentedColormap.from_list("custom_cmap", colors, N=len(label_names))

def make_background_transparent(image: np.ndarray) -> np.ndarray:
    background_mask = np.all(image[..., :3] == 0, axis=-1)  # Check if all color channels are 0
    image[..., -1] = np.where(background_mask, 0, 1)  # Set alpha based on the background_mask
    return image


def apply_colormap_to_label(label: np.ndarray, colormap: LinearSegmentedColormap) -> np.ndarray:
    """
    Applies a custom colormap to a label image.
    """
    # Ensure label is in the correct integer format
    label_img = label.astype(int)
    rgba_img = colormap(label_img)  # Map the label image through the colormap
    return rgba_img

labels_dict = {
        0: "фон",
        1: "селезёнка",
        2: "правая почка",
        3: "левая почка",
        4: "желчный пузырь",
        5: "пищевод",
        6: "печень",
        7: "желудок",
        8: "аорта",
        9: "поджелудочная железа",
        10: "надпочечник правый",
        11: "надпочечник левый",
        12: "кишечник",
    }

custom_cmap = create_colormap_for_labels(labels_dict, cmap_name='tab20')
 
def normalize(image: np.ndarray, 
                   min_val: int | None = None,
                   max_val: int | None = None):
    
    min_val = np.min(image) if min_val is None else min_val
    max_val = np.max(image) if max_val is None else max_val
    image = np.clip(image, min_val, max_val)

    image -= min_val
    image /= max_val - min_val
    return image


def vis_one(image: np.ndarray, 
            label: np.ndarray, 
            img_name: str,
            pred: np.ndarray | None = None,
            colormap: LinearSegmentedColormap=custom_cmap,
            soft_tissue_window: bool = False, 
            overlay: bool = False):
    figsize = 11 if pred is not None else 16
    plt.figure(figsize=(figsize, 5))
    
    num_subplots = 2 if pred is None else 3
    alpha = 0.5 if overlay else 1  # Transparency level

    # Image
    plt.subplot(1, num_subplots, 1)
    plt.xlabel("Image")
    if soft_tissue_window:
        image = normalize(image, -350, 400)
    plt.imshow(image, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(location='left', shrink=0.8)
    plt.title(img_name)
    
    # Label
    plt.subplot(1, num_subplots, 2)
    plt.xlabel("Label")
    plt.xticks([])
    plt.yticks([])
    # Apply and overlay colormap if enabled
    label_colored = apply_colormap_to_label(label, colormap)
    if overlay:
        plt.imshow(image, cmap="gray")
        label_colored = make_background_transparent(label_colored)
        plt.imshow(label_colored, alpha=1)
    else:
        plt.imshow(label_colored)

    if pred is not None:
        # Pred
        plt.subplot(1, num_subplots, 3)
        plt.xlabel("Pred")
        plt.xticks([])
        plt.yticks([])
        # Apply and overlay colormap if enabled
        pred_colored = apply_colormap_to_label(pred, colormap)
        if overlay:
            plt.imshow(image, cmap="gray")
            pred_colored = make_background_transparent(pred_colored)
            plt.imshow(pred_colored, alpha=1)
        else:
            pred_colored = apply_colormap_to_label(pred, colormap)
            plt.imshow(pred_colored)

    # Create legend
    labels_unique = np.unique(label)
    if pred is not None:
        labels_unique = np.union1d(labels_unique, np.unique(pred))

    if overlay:
        labels_unique = labels_unique[1:]

    handles = [Patch(color=colormap(i), label=name) for i, name in enumerate(labels_dict.values())
        if i in labels_unique]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

def img2label(img_path: str) -> str:
    return img_path.replace('images', 'labels').replace('img', 'lbl')
def load_npy(npy_path: str) -> np.ndarray:
    with open(npy_path, 'rb') as f:
        res = np.load(npy_path)
    return res

def vis_random_batch(data_root: str, num: int = 16, soft_tissue_window: bool = False):
    ls = os.listdir(os.path.join(data_root, 'images'))
    random.shuffle(ls)
    
    for item in ls[:num]:
        img_path = os.path.join(data_root, 'images', item)
        img = load_npy(img_path)
        lbl = load_npy(img2label(img_path))

        vis_one(img, lbl, item, soft_tissue_window)    

def plot_hist_with_legend(bars_dict: Dict):
    plt.bar(list(bars_dict.keys()), 
        list(bars_dict.values()),
             color=[custom_cmap(i) for i in bars_dict])
    legend_handles = [Patch(color=custom_cmap(i), label=labels_dict[i]) for i in bars_dict]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def vis_one_slice(sample_name: str):
    for i in range(1, 7):
        img_name = sample_name + f'_slice{i}_img.npy'
        img_path = os.path.join('data/default_dataset/images', img_name)
        img = load_npy(img_path)
        lbl = load_npy(img2label(img_path))

        vis_one(img, lbl, img_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def vis_predicts(model, dataset, n_samples, class_mapping, reverse_class_mapping, overlay=False):

    indices = range(len(dataset))
    indices = random.sample(indices, n_samples)
    with torch.no_grad():
        model.eval()
        for i in indices:
            image_path = dataset.images[i]
            image = load_npy(image_path)
            label = load_npy(img2label(image_path))
            label = class_mapping[label.astype(int)]
            label = reverse_class_mapping[label.astype(int)]
        
            image_tensor, _ = dataset[i]
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor.unsqueeze(0))
            outputs = F.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1).permute(0, 1, 2).squeeze(0)
            outputs = outputs.cpu().detach().numpy()
            outputs = reverse_class_mapping[outputs]
            outputs = cv2.resize(outputs, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)
            vis_one(image, label, 'test', pred=outputs, overlay=overlay)
