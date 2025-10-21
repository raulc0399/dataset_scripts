import os
import json
import tqdm

# Default prompts for different models
DEFAULT_PROMPTS = {
    "describe": "describe this image.",
    "modern_building": "does the image show a modern building? answer yes or no.",
    "detailed": "Describe this image in detail.",
}

def get_image_files(image_dir="./images"):
    """Get list of image files from directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    files = []
    for item in os.listdir(image_dir):
        item_path = os.path.join(image_dir, item)
        if os.path.isfile(item_path):
            _, ext = os.path.splitext(item.lower())
            if ext in image_extensions:
                files.append(item)
    return files

def load_or_create_metadata(image_files):
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

def process_images_with_prompts(image_files, metadata, process_func, model_name, image_dir="./images"):
    """Generic function to process images with any model using multiple prompts"""
    prompts = DEFAULT_PROMPTS

    # Process each prompt
    for prompt_key, prompt_text in prompts.items():
        caption_key = f"{model_name}_{prompt_key}_caption"
        print(f"\nProcessing with prompt '{prompt_key}': {prompt_text}")
        
        # Create a wrapper function that adds prompt info to the output
        def process_func_with_prompt_info(image_path, prompt):
            caption = process_func(image_path, prompt)
            return {
                "caption": caption,
                "prompt": prompt,
                "prompt_key": prompt_key
            }
        
        # Use existing process_images function
        process_images(image_files, metadata, process_func_with_prompt_info, caption_key, prompt_text, image_dir)

def process_images(image_files, metadata, process_func, caption_key, prompt=None, image_dir="./images"):
    """Generic function to process images with any model (backward compatibility)"""
    if prompt is None:
        prompt = DEFAULT_PROMPTS["describe"]

    # Process each image file
    for image_file in tqdm.tqdm(image_files):
        if image_file not in metadata:
            metadata[image_file] = {"file_name": image_file}
        
        # Skip if this model's caption already exists
        if caption_key in metadata[image_file]:
            print(f"Skipping {image_file} - {caption_key} already exists")
            continue
        
        image_path = os.path.join(image_dir, image_file)
        
        caption = process_func(image_path, prompt)
        print(f"Image: {image_file}, Caption: {caption}")

        metadata[image_file][caption_key] = caption
        
        # Save periodically
        if len([f for f in metadata if caption_key in metadata[f]]) % 10 == 0:
            save_metadata(metadata)

    save_metadata(metadata)
