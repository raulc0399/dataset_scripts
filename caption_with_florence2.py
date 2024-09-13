import os
import time
import argparse
from tqdm import tqdm
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
from transformers.dynamic_module_utils import get_imports

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

MODEL_ID = "microsoft/Florence-2-large"

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

def generate_image_caption(image_path, task_prompt = "<DETAILED_CAPTION>", text_input=None):
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

    caption_text = parsed_answer[task_prompt].replace("The image shows ", "")

    return caption_text

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate image captions with Florence2.")
    parser.add_argument("--write-results", action="store_true", default=False, help="Write results to results_florence2.txt")
    parser.add_argument("--image-dir", type=str, default="./test_images", help="Directory containing images to process")
    parser.add_argument("--trigger", type=str, default="", help="Trigger word or sentence for the caption generation")
    parser.add_argument("--test-run", action="store_true", default=False, help="Process only the first 10 images")
    args = parser.parse_args()

    # List image files in the directory
    image_dir = args.image_dir
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    # Limit the number of images if test_run is enabled
    if args.test_run:
        image_files = sorted(image_files)[:10]

    # Initialize variables for timing
    total_time = 0
    num_images = 0

    if args.write_results:
        f = open("results_florence2.txt", "w")
    else:
        f = None

    # Process each image
    print(f"Processing {len(image_files)} images...")

    for image_file in tqdm(image_files, desc="Processing images"):
        # print(f"Processing {image_file}...")

        image_path = os.path.join(image_dir, image_file)
        
        # Measure execution time
        start_time = time.time()
        caption = generate_image_caption(image_path)
        end_time = time.time()
        
        # Calculate and display execution time
        exec_time = end_time - start_time
        if f:
            f.write(f"--- Image: {image_file}, Execution Time: {exec_time:.2f} seconds\n")
            f.write(f"{caption}\n\n")
        
        # Save caption to a file with the same name but with .txt extension
        suffix = "_florence2" if args.test_run else ""
        caption_file_path = os.path.splitext(image_path)[0] + suffix + ".txt"
        with open(caption_file_path, "w") as caption_file:
            caption_file.write(f"{args.trigger} {caption}")
        
        # Accumulate total time and count
        total_time += exec_time
        num_images += 1

    # Calculate and display average execution time
    avg_time = total_time / num_images
    if f:
        f.write(f"Average Execution Time: {avg_time:.2f} seconds\n")
    if f:
        f.close()
