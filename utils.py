import os
import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

def create_colormap_for_labels(label_names: List[str], cmap_name: str = 'tab20') -> LinearSegmentedColormap:
    """
    Creates a colormap with specific colors mapped to label indices from a given colormap.
    """
    cmap = plt.get_cmap(cmap_name)  # Get the colormap
    # Create a color for each label based on its relative position in the label set
    colors = [cmap(i / len(label_names)) for i in range(len(label_names))]
    return LinearSegmentedColormap.from_list("custom_cmap", colors, N=len(label_names))

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
 

def vis_one(image: np.array, 
            label: np.array, 
            soft_tissue_window: bool = False, 
            colormap: LinearSegmentedColormap=custom_cmap):

    plt.figure(figsize=(10, 5))
    
    # Image
    plt.subplot(1, 2, 1)
    plt.title("Image")
    if soft_tissue_window:
        image = np.clip(image, -350, 400)
        image += 350
        image /= 750
    plt.imshow(image, cmap="gray")
    
    # Label
    plt.subplot(1, 2, 2)
    plt.title("Label")
    
    # Create and apply the custom colormap
    label_colored = apply_colormap_to_label(label, colormap)
    plt.imshow(label_colored)

    # Create legend
    labels_unique = np.unique(label)
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

        vis_one(img, lbl, soft_tissue_window)    