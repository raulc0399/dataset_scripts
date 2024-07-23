from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import os
import time

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
image_dir = "./test_images"
image_files = os.listdir(image_dir)

# Initialize variables for execution time and average time
total_time = 0
average_time = 0

with open("results_moondream.txt", "a") as f:
    # Process each image file
    print(f"Processing {len(image_files)} images...")

    for image_file in image_files:
        print(f"Processing {image_file}...")
        
        # Construct the full path to the image file
        image_path = os.path.join(image_dir, image_file)

        # Open the image
        image = Image.open(image_path)

        # Measure the execution time
        start_time = time.time()

        # Perform the image processing
        enc_image = model.encode_image(image)
        caption = model.answer_question(enc_image, "Describe this image.", tokenizer)

        # Calculate the execution time
        execution_time = time.time() - start_time

        # Print the result and execution time
    
        f.write(f"--- Image: {image_file}, Execution Time: {execution_time:.2f} seconds\n")
        f.write(f"{caption}\n\n")

        # Update the total time and average time
        total_time += execution_time

    average_time = total_time / len(image_files)

    # Print the average execution time
    f.write(f"Average execution time: {average_time} seconds\n")
