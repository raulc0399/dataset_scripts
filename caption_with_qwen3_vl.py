from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import tqdm
import json
import torch
from qwen_vl_utils import process_vision_info

model_id = "Qwen/Qwen3-VL-8B-Instruct-FP8"
# model_id = "Qwen/Qwen3-VL-8B-Thinking-FP8"
# model_id = "Qwen/Qwen3-VL-8B-Instruct"
# model_id = "Qwen/Qwen3-VL-8B-Thinking"
    
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
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
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("output text:", output_text)
    exit(0)

    return output_text.strip()

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
