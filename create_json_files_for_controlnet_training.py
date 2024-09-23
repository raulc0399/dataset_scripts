import os
import json
import argparse

# from metadata.jsonl, create json files for each image, as dataset for training controlnet with xflux
def create_json_files_from_metadata(folder_path):
    metadata_file = os.path.join(folder_path, 'metadata.jsonl')
    
    with open(metadata_file, 'r') as file:
        for line in file:
            entry = json.loads(line)
            image_name = entry['image']
            conditioning_image = entry['conditioning_image']
            text = entry['text']
            
            output_data = {
                'conditioning_image': conditioning_image,
                'caption': text
            }
            
            output_file_name = os.path.splitext(image_name)[0] + '.json'
            output_file_path = os.path.join(folder_path, output_file_name)
            
            with open(output_file_path, 'w') as output_file:
                json.dump(output_data, output_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create JSON files from metadata.jsonl for ControlNet training.')
    parser.add_argument('folder_path', type=str, help='The folder path containing metadata.jsonl')
    args = parser.parse_args()

    folder_path = args.folder_path
    create_json_files_from_metadata(folder_path)
