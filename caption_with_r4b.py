from transformers import AutoModel, AutoProcessor
from PIL import Image
import os
import tqdm
import json
import torch

model_id = "YannQi/R-4B"

model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
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
                    "image": "",
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

    print("chat template:", text)

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

def save_metadata(metadata):
    with open("metadata.jsonl", "a") as f:
        for item in metadata:
            json_line = json.dumps(item)
            f.write(json_line + '\n')

metadata = []

prompt = "Describe this image."

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
