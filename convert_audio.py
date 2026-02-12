import base64

# Change this to your audio filename
file_path = "sample.mp3"   # or sample.wav

with open(file_path, "rb") as audio_file:
    encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")

print("\nBase64 Output:\n")
print(encoded_string)

# Optional: save to txt file
with open("base64_output.txt", "w") as f:
    f.write(encoded_string)

print("\nSaved to base64_output.txt")
