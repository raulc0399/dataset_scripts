from transformers import AutoModel, AutoProcessor
import torch
from qwen_vl_utils import process_vision_info
from utils import get_image_files, load_or_create_metadata, process_images_with_prompts

# model_id = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
model_id = "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"

model = AutoModel.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
).to("cuda")

# Load processor
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def process_image(image_path: str, prompt: str, system_prompt: str = None) -> str:
    # Define conversation messages
    messages = []
    
    # Add system message if provided
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": prompt},
        ],
    })

    # Apply chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # print("chat template:", text)

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # print("output text:", output_text)

    return output_text[0].strip() if output_text else ""

if __name__ == "__main__":
    from utils import process_images_with_tasks
    
    image_files = get_image_files("./images")
    metadata = load_or_create_metadata(image_files)
    
    process_images_with_tasks(image_files, metadata, process_image, "llava_one_vision", image_dir="./images")
