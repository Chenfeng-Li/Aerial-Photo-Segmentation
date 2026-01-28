import sys
from pathlib import Path
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

# Previous directory
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))


from image_prediction import load_model, predict_image
from tools import plot_images_labels 

model_path = os.getenv("model_path", "checkpoints/best.pt")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chenfengli.com", "https://www.chenfengli.com"], # Domain of my personal website
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None 

@app.on_event("startup")
def _startup():
    """
    Load the model once when webapp starts.
    """
    global model
    model, _ = load_model(root/model_path)

@app.get("/health")
def health():
    """
    Health check.
    """
    return {"ok": True}

@app.post("/predict/image")
async def predict_image_api(file: UploadFile = File(...), combine: bool = False):
    """
    file (UploadFile):  Uploaded image.
    combine (bool): whether to combine the labels with original image.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    img_bytes = await file.read()

    # Use temp files as plot_images_labels expects file paths
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, file.filename or "input.png")
        out_path = os.path.join(td, "output.png")

        with open(in_path, "wb") as f:
            f.write(img_bytes)

        pred_lab = predict_image(model, in_path)
        plot_images_labels(in_path, pred_lab, combine, out_path) if combine else plot_images_labels(None, pred_lab, combine, out_path)

        if not os.path.exists(out_path):
            raise HTTPException(status_code=500, detail="Output image was not generated")

        with open(out_path, "rb") as f:
            out_bytes = f.read()

    return Response(content=out_bytes, media_type="image/png")