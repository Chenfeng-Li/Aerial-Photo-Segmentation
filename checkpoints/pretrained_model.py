from huggingface_hub import hf_hub_download
from pathlib import Path

path = Path(__file__).resolve().parent
hf_hub_download(
    repo_id="Chenfeng-Li/aerial-photo-segmentation",
    filename="best.pt",
    local_dir=path
)
print("Pretrained checkpoint best.pt downloaded.")
