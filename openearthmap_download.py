"""
Download and unzip the OpenEarthMap data.
Source: https://open-earth-map.org/overview.html
"""


import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    #Browser-like request
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    url = "https://zenodo.org/records/7223446/files/OpenEarthMap.zip?download=1"
    file_path = "OpenEarthMap.zip"

    # Download the file automatically. If failed, print manual download instruction.
    try:
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            progress_bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=file_path)

            with open(file_path, "wb") as f, progress_bar as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    except:
        print("Download failed. Please download manually at the following url:")
        print("https://zenodo.org/records/7223446")
        print("Unzip the file and put OpenEarthMap folder to the current directory.")


    # Unzip OpenEarthMap.zip and move the folder to current directory
    extract_dir = Path("tmp")
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(file_path, "r") as z:
        z.extractall(extract_dir)
    folder = list(extract_dir.iterdir())[0]
    target = Path.cwd() / folder.name
    shutil.move(str(folder), target)

    # Delete the zip file and temporary folder
    Path(file_path).unlink()
    shutil.rmtree(extract_dir)
