from pathlib import Path
from pathlib import PosixPath
from PIL import Image
import numpy as np
import argparse
import os
import sys

import torch
import segmentation_models_pytorch as smp

from tools import plot_images_labels

def load_model(model_path="checkpoints/best.pt"):
    """
    Load a model with checkpoint. Best model by default.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,  
        in_channels=3,
        classes=8,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint



def predict_image(model, img):
    """
    Given a model with an image, image array or directory, predict the label of every pixel
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    if isinstance(img, PosixPath) or type(img) is str:
        img = Image.open(img)
        img = img.convert('RGB')
    img_arr = np.array(img)

    # DeepLabV3+ required input be divisible by 16
    div = 16
    H, W, _ = img_arr.shape
    pad_h = (div - H % div) % div
    pad_w = (div - W % div) % div
    img_arr = np.pad(img_arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    x = (torch.from_numpy(img_arr).permute(2, 0, 1).float().unsqueeze(0)/255.0).to(device)  # (1,3,H,W)

    with torch.no_grad():
        logits = model(x)                 # (1,8,H,W)
        pred = logits.argmax(dim=1)       # (1,H,W)
    pred = pred.squeeze(0).cpu().numpy()  # (H,W), values 0..7
    pred = pred[:H, :W]

    return pred+1



if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default="")
    parser.add_argument('--combine', type=str, default="False")
    parser.add_argument('--save', type=str, default="")
    parser.add_argument('--model', type=str, default="checkpoints/best.pt")

    args = parser.parse_args()
    img = args.img
    combine = args.combine in ["True", "true", "TRUE"]
    save = args.save
    model_path = args.model

    if not os.path.isfile(img):
        print("Input image doesn't exists")
        sys.exit()

    model, _ = load_model(model_path)
    pred_lab = predict_image(model, img)
    plot_images_labels(img,pred_lab,combine,save)
