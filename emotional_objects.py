from transformers import pipeline
from PIL import Image

emotion_model = pipeline("image-classification", model="nateraw/vit-age-classifier")
aesthetic_model = pipeline("image-classification", model="cafeai/cafe_aesthetic")

image = Image.open("Chair2.png")

emo = emotion_model(image, top_k=2)
aesth = aesthetic_model(image, top_k=2)

print("Emotion:", emo)
print("Mood:", aesth)
