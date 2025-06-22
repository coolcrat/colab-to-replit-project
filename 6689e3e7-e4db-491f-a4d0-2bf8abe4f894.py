!pip install transformers diffusers torch accelerate matplotlib



# -------- Cell Separator --------

from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Text input
text = "I'm feeling calm and peaceful today."  # You can change this

# Step 2: Emotion Detection
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
result = classifier(text)[0][0]
emotion = result['label'].lower()
confidence = result['score']
print(f"✅ Detected Emotion: {emotion} ({confidence:.2f})")

# Step 3: Map to Art Prompt
emotion_to_prompt = {
    "joy": "A vibrant, sunny meadow full of flowers and butterflies",
    "sadness": "A rainy, grayscale cityscape with lonely streets",
    "anger": "A chaotic storm with red lightning in an abstract art style",
    "peace": "A calm lake with sunset and soft pastel tones",
    "fear": "A foggy, eerie forest with dark shadows",
    "love": "A romantic sunset over a hill with glowing stars"
}

prompt = emotion_to_prompt.get(emotion, "An expressive abstract painting of human emotion")
print(f"🎨 Generating image with prompt: '{prompt}'")

# Step 4: Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Step 5: Generate Image
image = pipe(prompt).images[0]

# Step 6: Show Image
plt.imshow(image)
plt.axis("off")
plt.title(f"Art for: {emotion}")
plt.show()


# -------- Cell Separator --------

image.save("generated_art.png")
from google.colab import files
files.download("generated_art.png")
