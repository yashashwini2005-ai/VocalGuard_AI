import base64
import os

folder = "dataset/Hindi/Non Human"

files = os.listdir(folder)

if not files:
    print("No files found in folder.")
    exit()

file_path = os.path.join(folder, files[0])

print("Using file:", file_path)

output_file = "real_base64.txt"

with open(file_path, "rb") as audio_file:
    encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")

with open(output_file, "w") as f:
    f.write(encoded_string)

print("Base64 saved successfully to:", output_file)
