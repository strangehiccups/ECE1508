# Download LJSpeech Dataset
import requests
import tarfile
import os

# Create data directory if it doesn't exist
os.makedirs("../data", exist_ok=True)

# URL for the official tar.bz2 dataset
url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
filename = "../data/LJSpeech-1.1.tar.bz2"

# Download with streaming
print("Downloading LJ Speech dataset...")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print("Download complete!")

# Extract
print("Extracting dataset...")
with tarfile.open(filename, "r:bz2") as tar:
    tar.extractall(path='../data')
print("Extraction complete!")
