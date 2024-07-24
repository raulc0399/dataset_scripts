from controlnet_aux.processor import OpenposeDetector
from PIL import Image

open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
open_pose = open_pose.to("cuda")

folder = "./test_images"

for i in range(10):
    file_name = f"{i+1}.jpg"
    print(f"Processing {file_name}")

    img = Image.open(f"{folder}/{file_name}").convert("RGB").resize((512, 512))
    processed_image_open_pose = open_pose(img, hand_and_face=True, detect_resolution=1024, image_resolution=1024)

    if processed_image_open_pose is None:
        print(f"No pose detected for {file_name}")
    else:
        processed_image_open_pose.save(f"{folder}/{i+1}_pose.jpg")