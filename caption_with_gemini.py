from google import genai
from google.genai import types
from PIL import Image
import os
import tqdm
import json

model_name = "gemini-2.5-flash"
# model_name = "gemini-2.5-pro"
client = genai.Client()

def process_image(image_path: str, prompt: str) -> str:
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Generate content with Gemini
        response = client.models.generate_content(
            model=model_name,
            contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            prompt
            ]
        )
        
        output_text = response.text
        print("output text:", output_text)
        
        return output_text.strip()
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

# List all files in the test_images directory
image_dir = "./images"
image_files = os.listdir(image_dir)

def save_metadata(metadata):
    with open("metadata.jsonl", "a") as f:
        for item in metadata:
            json_line = json.dumps(item)
            f.write(json_line + '\n')

metadata = []

prompt = "describe this image."

# Process each image file
for index, image_file in enumerate(tqdm.tqdm(image_files)):
    image_path = os.path.join(image_dir, image_file)
    
    caption = process_image(image_path, prompt)
    print(f"Image: {image_file}, Caption: {caption}")

    metadata.append({
        "file_name": image_file,
        "caption": caption
    })

    if index % 100 == 0:
        save_metadata(metadata)
        metadata = []
    
save_metadata(metadata)
