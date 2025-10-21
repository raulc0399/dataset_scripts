from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from utils import get_image_files, load_or_create_metadata, process_images_with_tasks

model_id = "Qwen/Qwen3-VL-8B-Instruct"
# model_id = "Qwen/Qwen3-VL-8B-Thinking"
    
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    # attn_implementation="flash_attention_2"
).to("cuda")

# Load processor
processor = AutoProcessor.from_pretrained(model_id)

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

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    # print("chat template:", inputs)
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # print("output text:", output_text)

    return output_text[0].strip() if output_text else ""

if __name__ == "__main__":
    image_files = get_image_files("./images")
    metadata = load_or_create_metadata(image_files)
    
    process_images_with_tasks(image_files, metadata, process_image, "qwen3_vl", image_dir="./images")
