import json
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm

def count_lines(file_path):
    with open(file_path, 'rb') as f:
        return sum(1 for _ in f)

def convert_json_to_parquet_with_images(json_file, parquet_file, images_folder):
    chunk_size = 150
    data = []

    # Calculate the maximum number of chunks
    total_lines = count_lines(json_file)
    max_nr = (total_lines + chunk_size - 1) // chunk_size

    # chunk_index is zero based, so max nr is one less    
    chunk_index = 0
    max_nr -= 1

    print(f"Converting {total_lines} lines to parquet with chunk size {chunk_size} and {max_nr} chunks")

    # Read JSON file and process in chunks
    with open(json_file, 'r') as f:
        for line in tqdm(f, desc="Reading JSON file"):
            data.append(json.loads(line))
            if len(data) == chunk_size:
                process_chunk(data, images_folder, parquet_file, chunk_index, max_nr)
                chunk_index += 1
                data = []

    # Process any remaining data
    if data:
        process_chunk(data, images_folder, parquet_file, chunk_index, max_nr)

def compress_to_png_bytes(image_path, quality=95):
    with Image.open(image_path) as img:
        # Convert to RGB if the image is in RGBA mode
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # Create a byte stream to save the compressed image
        byte_arr = io.BytesIO()
        
        # Save the image as PNG with compression
        img.save(byte_arr, format='PNG', optimize=True, quality=quality)
        
        # Return the byte data
        return byte_arr.getvalue()

def process_chunk(data, images_folder, parquet_file, chunk_index, max_nr):
    # Convert chunk to DataFrame
    df = pd.DataFrame(data)

    for column in ['image', 'conditioning_image']:
        df[column] = df[column].apply(lambda x: compress_to_png_bytes(os.path.join(images_folder, x)))

    # Create
    parquet_file_with_index = parquet_file.replace('.parquet', f'-{chunk_index:05d}-of-{max_nr:05d}.parquet')
    df.to_parquet(parquet_file_with_index, engine='pyarrow')

def convert_json_to_parquet(json_file, parquet_file):
    # Read JSON file
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    for column in ['image', 'conditioning_image']:
        df[column] = df[column].apply(lambda x: os.path.join("./images", x))

    df.to_parquet(parquet_file, engine='pyarrow')

# Usage
images_folder = '../persons-photo-concept-bucket-images-to-train/open_pose_controlnet/images'
json_file = '../persons-photo-concept-bucket-images-to-train/open_pose_controlnet/images/metadata.jsonl'
parquet_files_folder = '../persons-photo-concept-bucket-images-to-train/open_pose_controlnet/data'

os.makedirs(parquet_files_folder, exist_ok=True)

parquet_file = os.path.join(parquet_files_folder, 'train.parquet')

convert_json_to_parquet_with_images(json_file, parquet_file, images_folder)
# convert_json_to_parquet(json_file, parquet_file)
print(f"Conversion complete. Parquet files saved with pattern {parquet_file.replace('.parquet', '-<chunk_index>-of-<max_nr>.parquet')}")
