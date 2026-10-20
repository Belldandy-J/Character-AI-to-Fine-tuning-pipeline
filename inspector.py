import os
import json
from unsloth.chat_templates import get_chat_template
from transformers import AutoTokenizer

# --- 1. SETTINGS & TOKENIZER ---
local_model_dir = "./local_qwen2_5_vl_7b"

print(f"🔍 Loading offline tokenizer from {local_model_dir}...")
tokenizer = AutoTokenizer.from_pretrained(local_model_dir)

# Apply the exact same Unsloth ChatML template used in marathon.py
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
)

curriculum = [
    {"name": "memories", "path": "./memory_shard_harmony.jsonl"},
    {"name": "diaries",  "path": "./diary_shard_harmony.jsonl"},
    {"name": "chats",    "path": "./chat_shard_harmony.jsonl"}
]

# --- 2. INSPECTION LOOP ---
for stage in curriculum:
    print(f"\n{'='*50}")
    print(f"📂 INSPECTING SHARD: {stage['name'].upper()} ({stage['path']})")
    print(f"{'='*50}")

    if not os.path.exists(stage['path']):
        print(f"⚠️  File not found: {stage['path']} - Skipping.")
        continue

    # Load raw JSONL data
    with open(stage['path'], 'r', encoding='utf-8') as f:
        raw_dataset = [json.loads(line) for line in f]

    # 🔍 STEP 1: Extract System Prompt (Exact match to marathon.py)
    dynamic_system_prompt = ""
    for row in raw_dataset:
        if str(row.get("role", "")).lower() == "system":
            dynamic_system_prompt = str(row.get("content", ""))
            break

    if not dynamic_system_prompt:
        dynamic_system_prompt = "Continue the next sequence."
        print("⚠️  No native system prompt found. Using fallback.")
    else:
        print(f"🧠 System Prompt Extracted: '{dynamic_system_prompt[:60]}...'")

    # 🔨 STEP 2: Unified ChatML Formatting Logic (Mirroring marathon.py EXACTLY)
    token_counts = []
    visual_sample = None

    formatted_data = []
    current_conversation = []

    if dynamic_system_prompt:
        current_conversation.append({"role": "system", "content": dynamic_system_prompt})

    for row in raw_dataset:
        role = str(row.get("role", "")).lower()
        content = str(row.get("content", ""))

        if role in ["system", "developer"]:
            continue

        if role == "user":
            current_conversation.append({"role": "user", "content": content})

        elif role == "assistant":
            if len(current_conversation) == 0 or (len(current_conversation) == 1 and dynamic_system_prompt and current_conversation[0]["role"] == "system") or current_conversation[-1]["role"] != "user":
                current_conversation.append({"role": "user", "content": "Continue to encode our interactions."})

            turn_messages = [msg.copy() for msg in current_conversation]
            turn_messages.append({"role": "assistant", "content": content})

            # Apply chat template EXACTLY as marathon does
            text = tokenizer.apply_chat_template(turn_messages, tokenize=False, add_generation_prompt=False)

            # Tokenize the final formatted text block to get the true length Unsloth sees
            tokens = tokenizer(text)
            token_counts.append(len(tokens["input_ids"]))

            # Save the first sample for visual inspection
            if not visual_sample:
                visual_sample = text

            current_conversation.append({"role": "assistant", "content": content})

        # Reset context trigger EXACTLY like marathon
        if row.get("reset_context") or len(current_conversation) > 12:
            current_conversation = []
            if dynamic_system_prompt:
                current_conversation.append({"role": "system", "content": dynamic_system_prompt})

    # --- 3. REPORTING ---
    total_samples = len(token_counts)
    if total_samples == 0:
        print("⚠️  No valid samples found in this shard.")
        continue

    max_tokens = max(token_counts)
    min_tokens = min(token_counts)
    avg_tokens = sum(token_counts) // total_samples

    print(f"📊 STATS:")
    print(f"  Total Processed Samples : {total_samples}")
    print(f"  Shortest Sequence       : {min_tokens} tokens")
    print(f"  Average Sequence        : {avg_tokens} tokens")
    print(f"  🔥 MAX SEQUENCE LENGTH  : {max_tokens} tokens")

    print(f"\n👁️  VISUAL CHECK (First Sample Processed):")
    print("-" * 60)
    print(visual_sample)
    print("-" * 60)

print("\n✨ INSPECTION COMPLETE.")
