import os
import base64

INPUT_FOLDER = "evaluation_data/ai"   # change to real if needed
OUTPUT_FOLDER = "base64_files"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for file_name in os.listdir(INPUT_FOLDER):
    if file_name.endswith(".mp3") or file_name.endswith(".wav"):
        file_path = os.path.join(INPUT_FOLDER, file_name)

        with open(file_path, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")

        output_file = os.path.join(OUTPUT_FOLDER, file_name + ".txt")

        with open(output_file, "w") as f:
            f.write(encoded_string)

        print(f"Converted: {file_name}")

print("\nAll files converted successfully!")
