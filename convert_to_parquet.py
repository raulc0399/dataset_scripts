import json
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm

def convert_json_to_parquet_with_images(json_file, parquet_file, images_folder):
    chunk_size = 500
    data = []

    # Read JSON file and process in chunks
    with open(json_file, 'r') as f:
        for line in tqdm(f, desc="Reading JSON file"):
            data.append(json.loads(line))
            if len(data) == chunk_size:
                process_chunk(data, images_folder, parquet_file)
                data = []

    # Process any remaining data
    if data:
        process_chunk(data, images_folder, parquet_file)

def process_chunk(data, images_folder, parquet_file):
    # Convert chunk to DataFrame
    df = pd.DataFrame(data)

    for column in ['image', 'conditioning_image']:
        # df[column] = df[column].apply(lambda x: load_image(os.path.join(image_folder, x[0])))
        df[column] = df[column].apply(lambda x: Image.open(os.path.join(images_folder, x)).tobytes())

    # Create/Append to Parquet file
    if not os.path.exists(parquet_file):
        df.to_parquet(parquet_file, engine='fastparquet')
        exit()
    else:
        df.to_parquet(parquet_file, engine='fastparquet', append=True)

def convert_json_to_parquet(json_file, parquet_file):
    # Read JSON file
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    df.to_parquet(parquet_file, engine='pyarrow')

# Usage
images_folder = '../persons-photo-concept-bucket-images-to-train/open_pose_controlnet/images'
json_file = '../persons-photo-concept-bucket-images-to-train/open_pose_controlnet/images/metadata.jsonl'
parquet_file = '../persons-photo-concept-bucket-images-to-train/open_pose_controlnet/train.parquet'

# convert_json_to_parquet_with_images(json_file, parquet_file, images_folder)
convert_json_to_parquet(json_file, parquet_file)
print(f"Conversion complete. Parquet file saved as {parquet_file}")
