import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_ID = "microsoft/Florence-2-large"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

def generate_image_caption(image_path, task_prompt, text_input=None):
    image = Image.open(image_path)

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    return parsed_answer

if __name__ == "__main__":
    # List files in the directory
    image_dir = "./test_images"
    image_files = os.listdir(image_dir)

    # Initialize variables for timing
    total_time = 0
    num_images = 0

    with open("results_florence2.txt", "w") as f:
        # Process each image
        print(f"Processing {len(image_files)} images...")

        for image_file in image_files:
            print(f"Processing {image_file}...")

            image_path = os.path.join(image_dir, image_file)
            
            # Measure execution time
            start_time = time.time()
            caption = generate_image_caption(image_path, "<MORE_DETAILED_CAPTION>")
            end_time = time.time()
            
            # Calculate and display execution time
            exec_time = end_time - start_time
            f.write(f"--- Image: {image_file}, Execution Time: {exec_time:.2f} seconds\n")
            f.write(f"{caption}\n\n")
            
            # Accumulate total time and count
            total_time += exec_time
            num_images += 1

        # Calculate and display average execution time
        avg_time = total_time / num_images
        f.write(f"Average Execution Time: {avg_time:.2f} seconds\n")
