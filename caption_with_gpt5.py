import openai
from PIL import Image
import os
import tqdm
import json
import base64
from io import BytesIO

# model="o4-mini"
# model="gpt-4o-mini"
# model="gpt-5-mini"
model="gpt-5"

# Set up OpenAI client
client = openai.OpenAI()

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 string for OpenAI API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(image_path: str, prompt: str) -> str:
    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        output_text = response.choices[0].message.content
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
    output_dir = "images/output"
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    else:
        # Create base structure with all image files
        metadata = {}
        for image_file in image_files:
            metadata[image_file] = {"file_name": image_file}
        return metadata

def save_metadata(metadata):
    """Save metadata to JSON file"""
    output_dir = "images/output"
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

# Load or create metadata structure
metadata = load_or_create_metadata()

prompt = "describe this image."

# Process each image file
for image_file in tqdm.tqdm(image_files):
    if image_file not in metadata:
        metadata[image_file] = {"file_name": image_file}
    
    # Skip if this model's caption already exists
    if "gpt4o_caption" in metadata[image_file]:
        print(f"Skipping {image_file} - GPT-4o caption already exists")
        continue
    
    image_path = os.path.join(image_dir, image_file)
    
    caption = process_image(image_path, prompt)
    print(f"Image: {image_file}, Caption: {caption}")

    metadata[image_file]["gpt4o_caption"] = caption
    
    # Save periodically
    if len([f for f in metadata if "gpt4o_caption" in metadata[f]]) % 10 == 0:
        save_metadata(metadata)

save_metadata(metadata)
