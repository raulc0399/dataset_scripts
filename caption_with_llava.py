import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import os
import time
import argparse
from tqdm import tqdm
import re

max_new_tokens = 500

# https://arxiv.org/pdf/2310.00426.pdf, Fig. 10
prompt_for_caption = "Describe this image start with 'trigger_word', and be as descriptive as possible. describe the image in detail, including the objects, people, and actions in the image. "

def get_llava_next_model_and_processor():
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        # load_in_4bit=True
    ).to("cuda:0")

    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor

def generate_image_caption(image_path, trigger, model, processor):
    image = Image.open(image_path)
    # conversation = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": prompt_for_caption},
    #             {"type": "image"},
    #         ],
    #     },
    # ]
    # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    prompt=f"[INST] <image>\n{prompt_for_caption.replace('trigger_word', trigger)} [/INST]"

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return re.sub(r'\[INST\].*?\[/INST\]', '', caption, flags=re.DOTALL).strip()
        
if __name__ == "__main__":
    llava_model, llava_processor = get_llava_next_model_and_processor()
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate image captions with Llava.")
    parser.add_argument("--write-results", action="store_true", default=False, help="Write results to results_llava.txt")
    parser.add_argument("--image-dir", type=str, default="./test_images", help="Directory containing images to process")
    parser.add_argument("--trigger", type=str, default="", help="Trigger word or sentence for the caption generation")
    parser.add_argument("--test-run", action="store_true", default=False, help="Process only the first 10 images")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store the output text files")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
        f = open("results_llava.txt", "w")
    else:
        f = None

    # Process each image
    print(f"Processing {len(image_files)} images...")

    for image_file in tqdm(image_files, desc="Processing images"):
        # print(f"Processing {image_file}...")

        image_path = os.path.join(image_dir, image_file)
        
        # Save caption to a file with the same name but with .txt extension
        suffix = "_llava" if args.test_run else ""
        if args.output_dir:
            caption_file_path = os.path.join(args.output_dir, os.path.splitext(image_file)[0] + suffix + ".txt")
        else:
            caption_file_path = os.path.splitext(image_path)[0] + suffix + ".txt"

        # Check if caption file already exists
        if os.path.exists(caption_file_path):
            print(f"Caption file {caption_file_path} already exists. Skipping...")
            continue

        # Measure execution time
        start_time = time.time()
        caption = generate_image_caption(image_path, args.trigger, llava_model, llava_processor)
        end_time = time.time()

        # Calculate and display execution time
        exec_time = end_time - start_time
        if f:
            f.write(f"--- Image: {image_file}, Execution Time: {exec_time:.2f} seconds\n")
            f.write(f"{caption}\n\n")
            
        with open(caption_file_path, "w") as caption_file:
            caption_file.write(caption)
        
        # Accumulate total time and count
        total_time += exec_time
        num_images += 1

    # Calculate and display average execution time
    avg_time = total_time / num_images
    if f:
        f.write(f"Average Execution Time: {avg_time:.2f} seconds\n")
    if f:
        f.close()


