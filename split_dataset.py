import os, shutil, random

raw_imgs = [f for f in os.listdir("dataset/raw") if f.endswith(".jpg")]
random.shuffle(raw_imgs)
split = int(0.8 * len(raw_imgs))

for folder in ["dataset/images/train", "dataset/images/val",
               "dataset/labels/train", "dataset/labels/val"]:
    os.makedirs(folder, exist_ok=True)

for f in raw_imgs[:split]:
    shutil.copy(f"dataset/raw/{f}", f"dataset/images/train/{f}")
    shutil.copy(f"dataset/labels/{f.replace('.jpg','.txt')}",
                f"dataset/labels/train/{f.replace('.jpg','.txt')}")

for f in raw_imgs[split:]:
    shutil.copy(f"dataset/raw/{f}", f"dataset/images/val/{f}")
    shutil.copy(f"dataset/labels/{f.replace('.jpg','.txt')}",
                f"dataset/labels/val/{f.replace('.jpg','.txt')}")

print("Split done.")