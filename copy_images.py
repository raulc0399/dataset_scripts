from datasets import load_dataset
from tqdm import tqdm
import os

input_json_file_path = "../architecture_dataset.jsonl"

# Load the dataset
# dataset = load_dataset("ptx0/photo-concept-bucket")
dataset = load_dataset("json", data_files={"train": input_json_file_path})
print(dataset)

input_folder_name = "../photo-concept-bucket-images/images"
output_folder_name = "../architecture-photo-concept-bucket-images-to-train/images"

# Create a directory to store the images
os.makedirs(output_folder_name, exist_ok=True)

# Iterate through the dataset and download images
for i, item in enumerate(tqdm(dataset['train'])):
    file_name = f"image_{item['id']}.jpg"

    input_file_path = f"{input_folder_name}/{file_name}"
    output_file_path = f"{output_folder_name}/{file_name}"

    # Copy input to output file
    os.system(f"cp {input_file_path} {output_file_path}")
    

    