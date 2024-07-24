from datasets import load_dataset
import requests
from tqdm import tqdm
import os
import time

# Load the dataset
# dataset = load_dataset("ptx0/photo-concept-bucket")
dataset = load_dataset("json", data_files={"train": "../persons_dataset.jsonl"})
print(dataset)

folder_name = "../photo-concept-bucket-images"

# Create a directory to store the images
os.makedirs(folder_name, exist_ok=True)

# Iterate through the dataset and download images
for i, item in enumerate(tqdm(dataset['train'])):  # Assuming 'train' split, adjust if needed
    url = item['url']
    
    file_extension = url.split('.')[-1].replace('&fm=jpg', '')
    file_path = f"{folder_name}/image_{item['id']}.{file_extension}"

    if os.path.exists(file_path):
        print(f"Image {item['id']} already downloaded")
        continue    

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)

    except Exception as e:
        print(f"Error downloading image {item['id']}: {str(e)}")

    time.sleep(0.25)  # Be polite to the server
    