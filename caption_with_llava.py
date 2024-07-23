import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import os
import time

max_new_tokens = 500

# https://arxiv.org/pdf/2310.00426.pdf, Fig. 10
prompt_for_caption = "Describe this image and its style in a very detailed manner"

def get_llava_next_model_and_processor():
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
    
    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        # load_in_4bit=True
    ).to("cuda:0")

    return model, processor

def generate_image_caption(image_path, model, processor):
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

    prompt=f"[INST] <image>\n{prompt_for_caption} [/INST]"

    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
        
if __name__ == "__main__":
    llava_model, llava_processor = get_llava_next_model_and_processor()
    # List files in the directory
    image_dir = "./test_images"
    image_files = os.listdir(image_dir)

    # Initialize variables for timing
    total_time = 0
    num_images = 0

    with open("results_llava.txt", "w") as f:
        # Process each image
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            
            # Measure execution time
            start_time = time.time()
            caption = generate_image_caption(image_path, llava_model, llava_processor)
            end_time = time.time()
            
            # Calculate and display execution time
            exec_time = end_time - start_time
            print(f"--- Image: {image_file}, Execution Time: {exec_time:.2f} seconds")
            print(caption)
            print()
            
            # Accumulate total time and count
            total_time += exec_time
            num_images += 1

        # Calculate and display average execution time
        avg_time = total_time / num_images
        print(f"Average Execution Time: {avg_time:.2f} seconds")


