from google import genai
from google.genai import types
from PIL import Image
import os
import tqdm
import json

# model_name = "gemini-2.5-flash"
model_name = "gemini-2.5-pro"
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
        # print("output text:", output_text)
        
        return output_text.strip()
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

# List all files in the test_images directory
image_dir = "./images"
image_files = os.listdir(image_dir)

def load_or_create_metadata():
    """Load existing metadata or create new structure"""
    if os.path.exists("metadata.json"):
        with open("metadata.json", "r") as f:
            return json.load(f)
    else:
        # Create base structure with all image files
        metadata = {}
        for image_file in image_files:
            metadata[image_file] = {"file_name": image_file}
        return metadata

def save_metadata(metadata):
    """Save metadata to JSON file"""
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

# Load or create metadata structure
metadata = load_or_create_metadata()

prompt = "describe this image."

# Process each image file
for image_file in tqdm.tqdm(image_files):
    if image_file not in metadata:
        metadata[image_file] = {"file_name": image_file}
    
    # Skip if this model's caption already exists
    if "gemini_caption" in metadata[image_file]:
        print(f"Skipping {image_file} - Gemini caption already exists")
        continue
    
    image_path = os.path.join(image_dir, image_file)
    
    caption = process_image(image_path, prompt)
    print(f"Image: {image_file}, Caption: {caption}")

    metadata[image_file]["gemini_caption"] = caption
    
    # Save periodically
    if len([f for f in metadata if "gemini_caption" in metadata[f]]) % 10 == 0:
        save_metadata(metadata)

save_metadata(metadata)
