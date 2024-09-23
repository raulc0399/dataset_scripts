import os
import json
import argparse
from tqdm import tqdm

# from text files, create json files for each image, as dataset for training controlnet with xflux
def create_json_files_from_txt(folder_path):
    for file_name in tqdm(os.listdir(folder_path), desc="Processing text files"):
        if file_name.endswith('.txt'):
            txt_file_path = os.path.join(folder_path, file_name)
            with open(txt_file_path, 'r') as txt_file:
                caption = txt_file.read().strip()
            
            output_data = {
                'caption': caption
            }
            
            output_file_name = os.path.splitext(file_name)[0] + '.json'
            output_file_path = os.path.join(folder_path, output_file_name)
            
            with open(output_file_path, 'w') as output_file:
                json.dump(output_data, output_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create JSON files from text files for ControlNet training.')
    parser.add_argument('folder_path', type=str, help='The folder path containing text files')
    args = parser.parse_args()

    folder_path = args.folder_path
    create_json_files_from_txt(folder_path)
