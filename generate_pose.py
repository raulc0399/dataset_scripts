from controlnet_aux.processor import OpenposeDetector
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import os
import threading

class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.value += 1
            return self.value

    def get_value(self):
        with self.lock:
            return self.value

def process_image(file_queue, folder, pose_folder, open_pose_instance, pbar, counter):
    while True:
        try:
            file_name = file_queue.get_nowait()
        except Empty:
            break

        file_path = os.path.join(folder, file_name)

        output_file_name = f"{os.path.splitext(file_name)[0]}_pose.jpg"
        output_path = os.path.join(pose_folder, output_file_name)

        try:
            img = Image.open(file_path)
            processed_image_open_pose = open_pose_instance(img, hand_and_face=True, detect_resolution=1024, image_resolution=1024)
            if processed_image_open_pose is not None:
                processed_image_open_pose.save(output_path)
                os.system(f"cp {file_path} {output_path}")
                
                processed_count = counter.increment()
                pbar.set_description(f"With pose: {processed_count}")

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
        
        pbar.update(1)
        file_queue.task_done()

def main(folder, pose_folder, num_threads=8):
    files_to_process = []
    files_in_directory = os.listdir(folder)

    for file_name in files_in_directory:
        input_file_path = os.path.join(folder, file_name)

        if os.path.isfile(input_file_path) and input_file_path.lower().endswith(('.jpg', '.jpeg', '.png')) and not file_name.endswith("_pose.jpg"):
            files_to_process.append(file_name)

    file_queue = Queue()
    for file in files_to_process:
        file_queue.put(file)

    count_files = len(files_to_process)

    print(f"Processing {count_files} images")

    # Create instances of open_pose
    open_pose_instances = [OpenposeDetector.from_pretrained("lllyasviel/Annotators").to("cuda") for _ in range(num_threads)]

    print(f"Using {len(open_pose_instances)} threads")

    counter = Counter()
    with tqdm(total=count_files, desc="Processing images") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                future = executor.submit(
                    process_image,
                    file_queue,
                    folder,
                    pose_folder,
                    open_pose_instances[i],
                    pbar,
                    counter
                )
                futures.append(future)
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()

            print(f"Total processed images: {counter.get_value()}")

if __name__ == "__main__":
    # folder = "./test_images"
    folder = "../persons-photo-concept-bucket-images-to-train"
    pose_folder = os.path.join(folder, "pose")

    if not os.path.exists(pose_folder):
        os.makedirs(pose_folder)
        
    main(folder, pose_folder)
