import argparse
import os
from tqdm import tqdm

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate simple captions by saving the trigger.")
    parser.add_argument("--image-dir", type=str, default="./test_images", help="Directory containing images to process")
    parser.add_argument("--trigger", type=str, default="", help="Trigger word or sentence for the caption generation")
    parser.add_argument("--test-run", action="store_true", default=False, help="Process only the first 10 images")
    args = parser.parse_args()

    # List image files in the directory
    image_dir = args.image_dir
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

    # Limit the number of images if test_run is enabled
    if args.test_run:
        image_files = sorted(image_files)[:10]

    # Process each image
    print(f"Processing {len(image_files)} images...")

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        
        # Save trigger to a file with the same name but with .txt extension
        suffix = "_simple" if args.test_run else ""
        caption_file_path = os.path.splitext(image_path)[0] + suffix + ".txt"
        with open(caption_file_path, "w") as caption_file:
            caption_file.write(args.trigger)
