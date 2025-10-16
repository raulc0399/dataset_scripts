from transformers import AutoModel, AutoProcessor
from PIL import Image
import os
import tqdm
import json
import torch
from qwen_vl_utils import process_vision_info

# model_id = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
model_id = "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"

model = AutoModel.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
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
        add_generation_prompt=True
    )

    # print("chat template:", text)

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # print("output text:", output_text)

    return output_text[0].strip() if output_text else ""

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

prompt = "does the image show a modern building? answer yes or no."

# Process each image file
for image_file in tqdm.tqdm(image_files):
    if image_file not in metadata:
        metadata[image_file] = {"file_name": image_file}
    
    # Skip if this model's caption already exists
    if "llava_one_vision_caption" in metadata[image_file]:
        print(f"Skipping {image_file} - LLaVA-OneVision caption already exists")
        continue
    
    image_path = os.path.join(image_dir, image_file)
    
    caption = process_image(image_path, prompt)
    print(f"Image: {image_file}, Caption: {caption}")

    metadata[image_file]["llava_one_vision_caption"] = caption
    
    # Save periodically
    if len([f for f in metadata if "llava_one_vision_caption" in metadata[f]]) % 10 == 0:
        save_metadata(metadata)

save_metadata(metadata)
