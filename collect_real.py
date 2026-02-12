import os
import shutil

SOURCE = "Dataset/Real"   # your current real folder
DEST = "dataset/real"

os.makedirs(DEST, exist_ok=True)

count = 0
limit = 60   # number of real samples

for root, dirs, files in os.walk(SOURCE):
    for file in files:
        if file.endswith(".flac"):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(DEST, f"real_{count}.flac")
            shutil.copy(src_path, dst_path)
            count += 1

            if count >= limit:
                break
    if count >= limit:
        break

print("Copied", count, "real samples.")
