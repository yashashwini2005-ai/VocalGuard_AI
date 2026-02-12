import base64
import os

# 🔹 Change this to your actual file path
file_path = "dataset/real/61/70968/61-70968-0000.flac"

# 🔹 Output file name
output_file = "real_base64.txt"

# Check if file exists
if not os.path.exists(file_path):
    print("File not found:", file_path)
    exit()

with open(file_path, "rb") as audio_file:
    encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")

# Save to txt file
with open(output_file, "w") as f:
    f.write(encoded_string)

print("Base64 saved to:", output_file)

