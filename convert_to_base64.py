import base64
import os

# ======================================
# CHANGE THESE TWO PATHS
# ======================================

real_audio_path = r"C:\VocalGuard_AI\dataset\English\Real\real_sample.wav"
fake_audio_path = r"C:\VocalGuard_AI\dataset\English\Fake\fake_sample.wav"

# ======================================
# FUNCTION TO CONVERT
# ======================================

def convert_to_base64(file_path, output_name):

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")

    with open(output_name, "w", encoding="utf-8") as f:
        f.write(encoded_string)

    print(f"Base64 saved: {output_name}")


# ======================================
# CONVERT BOTH FILES
# ======================================

convert_to_base64(real_audio_path, "real_base64.txt")
convert_to_base64(fake_audio_path, "fake_base64.txt")

print("\nDone ✅")
