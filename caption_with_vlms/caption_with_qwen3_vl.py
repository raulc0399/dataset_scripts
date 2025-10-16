from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import tqdm
import json
import torch
from qwen_vl_utils import process_vision_info

model_id = "Qwen/Qwen3-VL-8B-Instruct"
# model_id = "Qwen/Qwen3-VL-8B-Thinking"
    
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    # attn_implementation="flash_attention_2"
).to("cuda")

# Load processor
processor = AutoProcessor.from_pretrained(model_id)

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

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    # print("chat template:", inputs)
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
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

prompt = "describe this image."

# Process each image file
for image_file in tqdm.tqdm(image_files):
    if image_file not in metadata:
        metadata[image_file] = {"file_name": image_file}
    
    # Skip if this model's caption already exists
    if "qwen3_vl_caption" in metadata[image_file]:
        print(f"Skipping {image_file} - Qwen3-VL caption already exists")
        continue
    
    image_path = os.path.join(image_dir, image_file)
    
    caption = process_image(image_path, prompt)
    print(f"Image: {image_file}, Caption: {caption}")

    metadata[image_file]["qwen3_vl_caption"] = caption
    
    # Save periodically
    if len([f for f in metadata if "qwen3_vl_caption" in metadata[f]]) % 10 == 0:
        save_metadata(metadata)

save_metadata(metadata)
