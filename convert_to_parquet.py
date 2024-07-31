import json
import pandas as pd
import os
from PIL import Image

def convert_json_to_parquet(json_file, parquet_file, images_folder):
    # Read JSON file
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    for column in ['image', 'conditioning_image']:
        # df[column] = df[column].apply(lambda x: load_image(os.path.join(image_folder, x[0])))
        df[column] = df[column].apply(lambda x: Image.open(os.path.join(images_folder, x)).tobytes())

    df.to_parquet(parquet_file, engine='pyarrow')

# Usage
images_folder = '../persons-photo-concept-bucket-images-to-train/open_pose_controlnet/images'
json_file = '../persons-photo-concept-bucket-images-to-train/open_pose_controlnet/images/metadata.jsonl'
parquet_file = '../persons-photo-concept-bucket-images-to-train/open_pose_controlnet/train.parquet'

convert_json_to_parquet(json_file, parquet_file, images_folder)
print(f"Conversion complete. Parquet file saved as {parquet_file}")