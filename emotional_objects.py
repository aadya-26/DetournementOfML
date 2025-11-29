from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont

# Load models
emotion_model = pipeline("image-classification", model="mo-thecreator/vit-Facial-Expression-Recognition")
aesthetic_model = pipeline("image-classification", model="cafeai/cafe_aesthetic")

# Load your image
image_path = "chair2.png"
image = Image.open(image_path)

# Run predictions
emo = emotion_model(image, top_k=3)
aesth = aesthetic_model(image, top_k=2)

# Remove "neutral" and pick next emotion
filtered_emo = [e for e in emo if e["label"].lower() != "neutral"]
dominant_emotion = filtered_emo[0]["label"] if filtered_emo else emo[0]["label"]

# Get aesthetic mood
dominant_aesthetic = aesth[0]["label"]

print("Emotion:", emo)
print("Filtered Emotion:", dominant_emotion)
print("Mood:", aesth)

# Prepare overlay text
text = f"Emotion: {dominant_emotion.upper()}\nMood: {dominant_aesthetic.upper()}"

# Draw overlay
draw = ImageDraw.Draw(image)

# Load a cute font (Comic Sans or Segoe Print)
try:
    font = ImageFont.truetype("ariblk.ttf", size=30)
except:
    try:
        font = ImageFont.truetype("ariblk.ttf", size=30)  # Segoe Print (Windows)
    except:
        font = ImageFont.load_default()

# Position — top-left corner
x, y = 40, 40

# Draw pink text overlay (hot pink)
draw.text((x, y), text, fill=(255, 105, 180), font=font)

# Display the image
image.show()

print(f"💖 Emotion: {dominant_emotion}")
print(f"🌸 Aesthetic: {dominant_aesthetic}")
