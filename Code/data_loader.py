import requests
import tarfile
import os

def download_ljspeech(url: str, path: str = "../data/LJSpeech-1.1"):
    if not os.path.exists(path):
        print("Downloading LJSpeech dataset...")
        lj_archive = f"{path}.tar.bz2"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(lj_archive, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete. Extracting...")
        with tarfile.open(lj_archive, "r:bz2") as tar:
            tar.extractall(path=path)
        os.remove(lj_archive)
        print("LJSpeech ready.")
    else:
        print("LJSpeech already present, skipping download.")

def download_librispeech(url: str, path: str = "../data/LibriSpeech", split: str = "test-clean"):
    if not os.path.exists(f"{path}/{split}"):
        print(f"Downloading LibriSpeech {split}...")
        ls_archive = f"{path}/LibriSpeech-{split}.tar.gz"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(ls_archive, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete. Extracting...")
        with tarfile.open(ls_archive, "r:gz") as tar:
            tar.extractall(path=path)
        os.remove(ls_archive)
        print(f"LibriSpeech {split} ready.")
    else:
        print(f"LibriSpeech {split} already present, skipping download.")