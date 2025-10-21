import os
import json
import tqdm

# Task configurations with system prompts and associated prompts
TASKS = {
    "architecture_detection": {
        "system_prompt": "You are an expert architectural analyst.",
        "prompts": {
            "p1": "Determine if the given image **depicts architecture (a building, structure, architectural feature)** Please answer with \"Yes\" or \"No\". Then briefly justify your answer (1–2 sentences).",
            "p2": "Rate from 0-10 how related this image is to architecture (buildings, structural design, interior spaces, architectural elements).",
        }
    },
    "image_quality": {
        "system_prompt": "You are a professional image quality assessor for architectural images.",
        "prompts": {
            "assess_quality": "Assess the overall quality of this image.",
            "check_composition": "Evaluate the composition and framing of this image.",
            "analyze_lighting": "Analyze the lighting conditions in this image."
        }
    },
    "architect_description": {
        "system_prompt": "You are a professional architect tasked with analyzing and describing architectural images for expert reference and dataset enrichment.",
        "prompts": {
            "describe_professionally": "Provide a professional architectural description of this image.",
            "analyze_materials": "Analyze the materials and construction techniques visible in this image.",
            "evaluate_design": "Evaluate the design principles demonstrated in this architectural image."
        }
    }
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

def process_images_with_tasks(image_files, metadata, process_func, model_name, image_dir="./images"):
    """Generic function to process images with any model using task-based prompts"""
    tasks = TASKS
    
    # Process each task
    for task_name, task_config in tasks.items():
        system_prompt = task_config["system_prompt"]
        prompts = task_config["prompts"]
        
        print(f"\n=== Processing task: {task_name} ===")
        print(f"System prompt: {system_prompt}")
        
        # Process each prompt within the task
        for prompt_key, prompt_text in prompts.items():
            caption_key = f"{model_name}_{task_name}_{prompt_key}_caption"
            print(f"\nProcessing prompt '{prompt_key}': {prompt_text}")
            
            # Use existing process_images function
            process_images(image_files, metadata, process_func, caption_key, prompt_text, image_dir, system_prompt)

def process_images(image_files, metadata, process_func, caption_key, prompt, image_dir="./images", system_prompt=None):
    # Process each image file
    for image_file in tqdm.tqdm(image_files):
        if image_file not in metadata:
            metadata[image_file] = {"file_name": image_file}
        
        # Skip if this model's caption already exists
        if caption_key in metadata[image_file]:
            print(f"Skipping {image_file} - {caption_key} already exists")
            continue
        
        image_path = os.path.join(image_dir, image_file)
        
        result = process_func(image_path, prompt, system_prompt)
        metadata[image_file][caption_key] = result
                    
        print(f"Image: {image_file}, Caption: {result}")

        # Save periodically
        if len([f for f in metadata if caption_key in metadata[f]]) % 10 == 0:
            save_metadata(metadata)

    save_metadata(metadata)
