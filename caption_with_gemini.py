import google.generativeai as genai
from PIL import Image
import os
import tqdm
import json

# Configure Gemini
model_name = "gemini-2.0-flash-exp"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel(model_name)

def process_image(image_path: str, prompt: str) -> str:
    try:
        # Load image
        image = Image.open(image_path)
        
        # Generate content with Gemini
        response = model.generate_content([prompt, image])
        
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
