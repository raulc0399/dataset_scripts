from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os
import tqdm
import json

model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,
    device_map={"": "cuda"},
    # attn_implementation="flash_attention_2"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model.generation_config.pad_token_id = tokenizer.pad_token_id

# List all files in the test_images directory
image_dir = "./images"
image_files = os.listdir(image_dir)

def save_metadata(metadata):
    with open("metadata.jsonl", "a") as f:
        for item in metadata:
            json_line = json.dumps(item)
            f.write(json_line + '\n')

metadata = []

# Process each image file
for index, image_file in enumerate(tqdm.tqdm(image_files)):
    # Construct the full path to the image file
    image_path = os.path.join(image_dir, image_file)

    # Open the image
    image = Image.open(image_path)

    # Perform the image processing
    enc_image = model.encode_image(image)
    caption = model.answer_question(enc_image, "Describe this image.", tokenizer)

    metadata.append({
        "file_name": image_file,
        "caption": caption
    })

    if index % 100 == 0:
        save_metadata(metadata)
        metadata = []
    
save_metadata(metadata)