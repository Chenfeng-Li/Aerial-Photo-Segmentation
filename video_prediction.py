import numpy as np
import cv2
import argparse
from tqdm import tqdm
import os
import sys

# Fix plotting issue for macbook
import matplotlib
matplotlib.use("Agg")

from tools import plot_images_labels
from image_prediction import load_model, predict_image

def predict_video(model, video, save, combine=False):
    """
    For a video, predict frame by frame and output the video.
    """
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = cap.get(cv2.CAP_PROP_FPS)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save, fourcc, out_fps, (w, h), True)
    frame_idx = 0

    try:
        with tqdm(total=total_frames, desc="Labeling video", unit="frame") as pbar:
            while True:
                ret, frame_bgr = cap.read()
                if not ret: #End of video
                    break 

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pred = predict_image(model, frame_rgb)
                overlay_rgb = plot_images_labels(frame_rgb, pred, combine, return_array=True) if combine else plot_images_labels(None, pred, combine, return_array=True)
                if overlay_rgb.dtype != np.uint8:
                    overlay_rgb = np.clip(overlay_rgb, 0, 255).astype(np.uint8)

                overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                writer.write(overlay_bgr)
                frame_idx += 1
                pbar.update(1)
    finally:
        cap.release()
        writer.release()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default="")
    parser.add_argument('--combine', type=str, default="False")
    parser.add_argument('--save', type=str, default="predict.mp4")
    parser.add_argument('--model', type=str, default="checkpoints/best.pt")

    args = parser.parse_args()
    video = args.video
    combine = args.combine in ["True", "true", "TRUE"]
    save = args.save
    model_path = args.model

    if not os.path.isfile(video):
        print("Input video doesn't exists")
        sys.exit()
    
    model, _ = load_model(model_path)
    predict_video(model, video, save, combine)
