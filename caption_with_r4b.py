from transformers import AutoModel, AutoProcessor
from PIL import Image
import os
import tqdm
import json
import torch

model_id = "YannQi/R-4B"

model = AutoModel.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
).to("cuda")

# Load processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def process_image(image_path: str, prompt: str) -> str:
    # Define conversation messages
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        thinking_mode="auto"
    )

    # print("chat template:", text)

    image = Image.open(image_path)

    # Process inputs
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt"
    ).to("cuda")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=16384)
    output_ids = generated_ids[0][len(inputs.input_ids[0]):]

    # Decode output
    output_text = processor.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return output_text.strip()

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

prompt = "Describe this image."

# Process each image file
for image_file in tqdm.tqdm(image_files):
    if image_file not in metadata:
        metadata[image_file] = {"file_name": image_file}
    
    # Skip if this model's caption already exists
    if "r4b_caption" in metadata[image_file]:
        print(f"Skipping {image_file} - R4B caption already exists")
        continue
    
    image_path = os.path.join(image_dir, image_file)
    
    caption = process_image(image_path, prompt)
    print(f"Image: {image_file}, Caption: {caption}")

    metadata[image_file]["r4b_caption"] = caption
    
    # Save periodically
    if len([f for f in metadata if "r4b_caption" in metadata[f]]) % 10 == 0:
        save_metadata(metadata)

save_metadata(metadata)
