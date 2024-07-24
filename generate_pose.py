from controlnet_aux.processor import OpenposeDetector
from PIL import Image

open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

folder = "./test_images"

for i in range(10):
    img = Image.open(f"{folder}/{i+1}.jpg").convert("RGB").resize((512, 512))
    processed_image_open_pose = open_pose(img, hand_and_face=True, detect_resolution=1024, image_resolution=1024)

    processed_image_open_pose.save(f"{folder}/{i+1}_pose.jpg")