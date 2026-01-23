"""
Tool functions
"""
from pathlib import Path
from pathlib import PosixPath
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

import random


def dataset_structure(folder=Path("OpenEarthMap_wo_xBD"), islabel=True):
    """
    Given the directory of openearthmap, return the list of city name, the directory of each images and the corresponding label.
    folder (Path): Directory of the OpenEarthMap folder.
    islabel (bool): If True, only return the images with labels and the corresponding label; if False, only return the image with no label.
    """
    cities, images, labels = [], [], []
    for path in folder.iterdir():
        if path.is_dir():
            cities.append(path.name)
            img_dir = path/"images"
            lab_dir = path/"labels"
            for img_path in img_dir.iterdir():
                lab_path = lab_dir/img_path.name
                if islabel:
                    if lab_path in lab_dir.iterdir():
                        images.append(img_path)
                        labels.append(lab_path)
                else:
                    if lab_path not in lab_dir.iterdir():
                        images.append(img_path)

    return [cities, images, labels] if islabel else [cities, images, None]





# Color map defination, following OpenEarthMap's color map
classes = ["invalid", "bareland", "rangeland", "developed space", "road", "tree", "water", "agriculture land", "building"]
hex_colors = ["#000000", "#800000", "#00FF24", "#949494", "#FFFFFF", "#226126", "#0045FF",  "#4BB549",  "#DE1F07"]
cmap = ListedColormap(hex_colors)
n_classes = len(classes)
norm = BoundaryNorm(boundaries=np.arange(-0.5, n_classes + 0.5, 1), ncolors=n_classes)
legend_patches = [Patch(facecolor=hex_colors[i], label=classes[i]) for i in range(len(classes))]
legend_patches_transparent = [Patch(facecolor=hex_colors[i], label=classes[i], alpha=0.4) for i in range(len(classes))]

def plot_images_labels(img=None, lab=None, combine=False, save_dir=""):

    """
    Display the images and/or corresponding labels
    img (PIL image, np array, Path, str or None): An image, image array or the directory of an image. If None then will not be print.
    lab (PIL image, np array, Path, String or None): An label, label array or the directory of an label. If None then will not be print.
    combine (bool): if False, plot the image and label side by side; if True, plot the image with transparent label on it.
    save_dir (Path, str or None): If not None, save the plot to the directory.
    """
    if isinstance(img, PosixPath) or type(img) is str:
        img = Image.open(img)
        img = img.convert('RGB')
    if isinstance(lab, PosixPath) or type(lab) is str:
        lab = Image.open(lab)
        lab = lab.convert('L')

        
    if img is None and lab is None:
        return

    elif lab is None:
        plt.imshow(img)
        plt.axis("off")

    elif img is None:
        lab_arr = np.array(lab)
        plt.imshow(lab_arr, cmap=cmap, norm=norm, interpolation="nearest")
        plt.axis("off")
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0,)

    else:
        lab_arr = np.array(lab)
        if not combine:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img)
            axes[0].set_title("Image")
            axes[1].imshow(lab_arr, cmap=cmap, norm=norm, interpolation="nearest")
            axes[1].set_title("Label")
            axes[1].axis("off")
            axes[1].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
            plt.tight_layout()
            
        else:
            plt.imshow(img)
            plt.imshow(lab_arr, cmap=cmap, interpolation="nearest", norm=norm, alpha=0.4)
            plt.legend(handles=legend_patches_transparent, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
            plt.axis("off")
        

    if save_dir:
        plt.savefig(save_dir, bbox_inches="tight")
    plt.show()
        


def pad_to_min_size(img_arr, lab_arr=None, min_h=512, min_w=512):
    """
    Pad the image and corresponding label (if any) to minimum size. Pads on bottom and/or right.
    """
    h, w, _ = img_arr.shape
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)

    if pad_h == 0 and pad_w == 0:
        return img_arr, lab_arr
    img_arr = np.pad(img_arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    lab_arr = np.pad(lab_arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0) if lab_arr is not None else None
    return img_arr, lab_arr

def random_crop(img_arr, lab_arr=None, crop_h=512, crop_w=512):
    """
    Randomly crop an image and corresponding label (if any) to a specific size.
    """
    h, w, _ = img_arr.shape
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    img_arr = img_arr[top:top+crop_h, left:left+crop_w, :]
    lab_arr = lab_arr[top:top+crop_h, left:left+crop_w] if lab_arr is not None else None
    return img_arr, lab_arr


def center_crop(img_arr, lab_arr=None, crop_h=512, crop_w=512):
    """
    Centered crop an image and corresponding label (if any) to a specific size. 
    """
    h, w, _ = img_arr.shape
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    img_arr = img_arr[top:top+crop_h, left:left+crop_w, :]
    lab_arr = lab_arr[top:top+crop_h, left:left+crop_w] if lab_arr is not None else None
    return img_arr, lab_arr

def remap_labels(lab_arr, IGNORE = 255):
    """
    The label are 0-8, where 0 is invalid and 1-8 are valid classification.
    Change 0 to an IGNORE label and set valid label to 0-7.
    """
    lab = lab_arr.astype(np.int64)
    lab[lab >= 1] -= 1
    lab[lab == 0] = IGNORE
    return lab



if __name__=="__main__":
    pass
