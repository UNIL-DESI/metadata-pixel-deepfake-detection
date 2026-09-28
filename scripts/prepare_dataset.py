import os
import zipfile

out_dir = "data/raw/Hybrid_Deepfake_Dataset"
os.makedirs(out_dir, exist_ok=True)

# 1. Extract Celeb-DF and DFDC from Deepfakes.zip
if os.path.exists("Deepfakes.zip"):
    print("Extracting Celeb-DF and DFDC from Deepfakes.zip...")
    with zipfile.ZipFile("Deepfakes.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            file_lower = file.lower()
            if (
                "celeb-df" in file_lower
                or "celeb_df" in file_lower
                or "dfdc" in file_lower
            ):
                zip_ref.extract(file, out_dir)
    os.remove("Deepfakes.zip")

# 2. Extract all contents from DeeperForensics.zip
if os.path.exists("DeeperForensics.zip"):
    print("Extracting all contents from DeeperForensics.zip...")
    with zipfile.ZipFile("DeeperForensics.zip", "r") as zip_ref:
        zip_ref.extractall(out_dir)
    os.remove("DeeperForensics.zip")

print("Dataset preparation complete!")