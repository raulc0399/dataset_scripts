from controlnet_aux.processor import OpenposeDetector
from PIL import Image
import os
from tqdm import tqdm

open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
open_pose = open_pose.to("cuda")

folder = "./test_images"
# folder = "../persons-photo-concept-bucket-images-to-train"
pose_folder = os.path.join(folder, "pose")

if not os.path.exists(pose_folder):
    os.makedirs(pose_folder)

files = os.listdir(folder)

for file_name in tqdm(files):
    input_file_path = os.path.join(folder, file_name)
    if os.path.isdir(input_file_path) or not input_file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    # print(f"Processing {file_name}")

    copied_file_path = os.path.join(pose_folder, file_name)
                                    
    img = Image.open(input_file_path)
    processed_image_open_pose = open_pose(img, hand_and_face=True, detect_resolution=1024, image_resolution=1024)

    if processed_image_open_pose is None:
        # print(f"No pose detected for {file_name}")
        pass
    else:
        os.system(f"cp {input_file_path} {copied_file_path}")
        processed_image_open_pose.save(os.path.join(pose_folder, f"{os.path.splitext(file_name)[0]}_pose.jpg"))