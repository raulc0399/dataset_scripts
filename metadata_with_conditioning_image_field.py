import os
import json
from tqdm import tqdm

def main():
    base_folder = "../persons-photo-concept-bucket-images-to-train"
    pose_filtered_folder = os.path.join(base_folder, "pose_filtered")
    input_jsonl = os.path.join(pose_filtered_folder, "metadata_orig.jsonl")
    output_jsonl = os.path.join(pose_filtered_folder, "metadata.jsonl")

    # Create a set of existing file IDs for fast lookup
    existing_files = set(
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir(pose_filtered_folder)
        if f.startswith('image_') and f.endswith('.jpg') and not f.endswith('_pose.jpg')
    )

    # Process the JSONL file
    with open(input_jsonl, 'r') as in_file, open(output_jsonl, 'w') as out_file:
        for line in tqdm(in_file, desc="Filtering JSONL"):
            entry = json.loads(line)
            if entry['id'] in existing_files:
                out_file.write(json.dumps({
                    'id': entry['id'],
                    'image': f"image_{entry['id']}.jpg",
                    'conditioning_image': f"image_{entry['id']}_pose.jpg",
                    'text': entry['cogvlm_caption']
                }) + '\n')
            else:
                print(f"Missing file for {entry['id']}")

    print(f"metadata JSONL saved to {output_jsonl}")

if __name__ == "__main__":
    main()