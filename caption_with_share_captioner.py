from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time

model_name = "Lin-Chen/ShareCaptioner"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True).eval()
model.tokenizer = tokenizer

model.cuda()

seg1 = '<|User|>:'
seg2 = f'Analyze the image in a comprehensive and detailed manner.{model.eoh}\n<|Bot|>:'
seg_emb1 = model.encode_text(seg1, add_special_tokens=True).cuda()
seg_emb2 = model.encode_text(seg2, add_special_tokens=False).cuda()

def detailed_caption(img_path):
    subs = []
    image = Image.open(img_path).convert("RGB")
    subs.append(model.vis_processor(image).unsqueeze(0))

    subs = torch.cat(subs, dim=0).cuda()
    tmp_bs = subs.shape[0]
    tmp_seg_emb1 = seg_emb1.repeat(tmp_bs, 1, 1)
    tmp_seg_emb2 = seg_emb2.repeat(tmp_bs, 1, 1)
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            subs = model.encode_img(subs)
            input_emb = torch.cat([tmp_seg_emb1, subs, tmp_seg_emb2], dim=1)
            out_embeds = model.internlm_model.generate(inputs_embeds=input_emb,
                                                       max_length=500,
                                                       num_beams=3,
                                                       min_length=1,
                                                       do_sample=True,
                                                       repetition_penalty=1.5,
                                                       length_penalty=1.0,
                                                       temperature=1.,
                                                       eos_token_id=model.tokenizer.eos_token_id,
                                                       num_return_sequences=1,
                                                       )

    return model.decode_text([out_embeds[0]])

if __name__ == "__main__":
    # Directory containing the test images
    image_dir = "./test_images"

    # Get a list of all files in the directory
    image_files = os.listdir(image_dir)

    # Initialize variables for total time and number of files
    total_time = 0
    num_files = 0

    # Open the file to write the results
    with open("results_share_captioner.txt", "w") as f:
        # Iterate over the files
        print(f"Processing {len(image_files)} images...")
        
        for image_file in image_files:
            print(f"Processing {image_file}...")

            # Get the path to the image file
            img_path = os.path.join(image_dir, image_file)
            
            # Start the timer
            start_time = time.time()
            
            # Generate the detailed caption for the image
            caption = detailed_caption(img_path)
            
            # Calculate the execution time
            execution_time = time.time() - start_time
            
            # Write the caption and execution time to the file
            f.write(f"--- Image: {image_file}, Execution Time: {execution_time:.2f} seconds\n")
            f.write(f"{caption}\n\n")
            
            # Update the total time and number of files
            total_time += execution_time
            num_files += 1

        # Calculate the average time
        average_time = total_time / num_files

        # Write the average time to the file
        f.write(f"Average Time: {average_time} seconds\n")
