import os
import json
from tqdm import tqdm

def main():
    base_folder = "../persons-photo-concept-bucket-images-to-train/"
    pose_filtered_folder = os.path.join(base_folder, "pose_filtered")
    input_jsonl = os.path.join(base_folder, "persons_dataset_pose_filtered.jsonl")
    output_jsonl = os.path.join(base_folder, "persons_dataset_pose_filtered_updated.jsonl")

    # Create a set of existing file IDs for fast lookup
    existing_files = set(
        int(f.split('_')[1].split('.')[0])
        for f in os.listdir(pose_filtered_folder)
        if f.startswith('image_') and f.endswith('.jpg')
    )

    # Process the JSONL file
    with open(input_jsonl, 'r') as in_file, open(output_jsonl, 'w') as out_file:
        for line in tqdm(in_file, desc="Filtering JSONL"):
            entry = json.loads(line)
            if entry['id'] in existing_files:
                out_file.write(line)

    print(f"Filtered JSONL saved to {output_jsonl}")

if __name__ == "__main__":
    main()