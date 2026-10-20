from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

base_path = "/path/to/your/local_qwen2_5_vl_7b"
lora_path = "/path/to/your/dora_lora_after_chats"
save_path = "/path/to/your/character-qwen2_5_vl-bf16"

print("1. Loading raw Qwen base model to CPU RAM...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_path,
    torch_dtype=torch.float32, # MUST be float32 for Zen 3 CPU math
    device_map="cpu"
)

print("2. Mounting DoRA adapter...")
model = PeftModel.from_pretrained(base_model, lora_path)

print("3. Using PEFT to permanently fuse DoRA magnitudes into base weights...")
model = model.merge_and_unload()

print("4. Downcasting the mathematically safe model back to bfloat16...")
model = model.to(torch.bfloat16)

print("5. Saving the merged model...")
model.save_pretrained(save_path)

processor = AutoProcessor.from_pretrained(base_path)
processor.save_pretrained(save_path)

print(f"Merge complete! Saved to: {save_path}")
