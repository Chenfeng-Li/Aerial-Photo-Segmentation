from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Chenfeng-Li/aerial-photo-segmentation",
    filename="best.pt",
    local_dir_use_symlinks=False
)
print("Pretrained checkpoint best.pt downloaded.")
