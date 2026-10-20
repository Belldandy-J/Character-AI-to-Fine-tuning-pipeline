import os
# Force PyTorch to dynamically expand memory segments to prevent fragmentation OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import torch
import datetime
import time
import json

from unsloth.chat_templates import get_chat_template
from unsloth import FastVisionModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from transformers.trainer_callback import PrinterCallback
from huggingface_hub import snapshot_download

# --- 1. SETTINGS, OFFLINE BUNKER & UNIVERSAL CURRICULUM ---
model_name = "unsloth/Qwen2.5-VL-7B-Instruct" # Target the 7B Vision Model
max_seq_length = 4096
local_model_dir = "./local_qwen2_5_vl_7b"

# Securely download the model locally first if it doesn't exist
if not os.path.exists(local_model_dir):
    print(f"🌐 Local model not found. Downloading '{model_name}' to working directory...")
    snapshot_download(
        repo_id = model_name,
        local_dir = local_model_dir,
        local_dir_use_symlinks = False,
        ignore_patterns = ["*.msgpack", "*.h5", "*.ot"]
    )
    print("✅ Download complete! Cutting internet dependencies now.")
else:
    print(f"📦 Found cached local model at: {local_model_dir}")

# Hard-lock environment variables to block ALL web calls
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Blind PyTorch to Amber's GPU so we don't mix RDNA architectures
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["ROCR_VISIBLE_DEVICES"] = "0"

# Redirect the pointer to our local offline directory
model_name = local_model_dir

# Universal shard settings
# Epochs, LRs softened for 7B continuous fine-tuning
curriculum = [
    {"name": "memories", "path": "./memory_shard_harmony.jsonl", "epochs": 3, "max_seq": 2048, "optim": "paged_adamw_8bit", "learning_rate": 2e-5, "pack": False},
    {"name": "diaries",  "path": "./diary_shard_harmony.jsonl",  "epochs": 3, "max_seq": 2048, "optim": "paged_adamw_8bit", "learning_rate": 5e-5, "pack": False},
    {"name": "chats",    "path": "./chat_shard_harmony.jsonl",   "epochs": 3, "max_seq": 2560, "optim": "paged_adamw_8bit", "learning_rate": 2e-5, "pack": False}
]

# --- 2. LOGGING SETUP ---
def get_next_log_file():
    run_idx = 1
    while True:
        log_name = f"Unsloth-Qwen-Training-Log-run#{run_idx:03d}.txt"
        if not os.path.exists(log_name):
            return log_name
        run_idx += 1

log_file_path = get_next_log_file()
print(f"📝 Initializing fresh log file: {log_file_path}")

def log_to_file(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(formatted_message + "\n")

class FancyProgressCallback(TrainerCallback):
    def __init__(self, stage_name):
        self.stage_name = stage_name
        self.start_time = None
        self.start_step = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.start_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = logs["loss"]
            try:
                perplexity = math.exp(loss) if loss < 20 else float('inf')
            except OverflowError:
                perplexity = float('inf')

            step = state.global_step
            max_steps = state.max_steps
            epoch = logs.get("epoch", 0.0)
            lr = logs.get("learning_rate", 0.0)
            grad_norm = logs.get("grad_norm", 0.0)

            # Cumulative ETA Calculation
            if self.start_time is None:
                self.start_time = time.time()
                self.start_step = step

            steps_completed = step - self.start_step

            if steps_completed > 0:
                elapsed_time = time.time() - self.start_time
                time_per_step = elapsed_time / steps_completed
                remaining_steps = max_steps - step
                seconds_remaining = time_per_step * remaining_steps

                hours, rem = divmod(int(seconds_remaining), 3600)
                minutes, seconds = divmod(rem, 60)
                eta_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"  # Renders as 12:59:30
                # Alternatively: eta_str = f"{hours}h {minutes:02d}m" # Renders as 12h 59m
            else:
                eta_str = "--:--:--"

            progress_pct = (step / max_steps) * 100 if max_steps > 0 else 0

            bar_length = 20
            filled_length = int(bar_length * step // max_steps) if max_steps > 0 else 0
            bar = '█' * filled_length + ' ' * (bar_length - filled_length)

            log_line = (
                f"[{self.stage_name.upper()}] | Stage ETA: {eta_str} | Stage Progress: {progress_pct:6.2f}% |{bar}| "
                f"Step: {step:5d}/{max_steps:5d} | "
                f"Loss: {loss:.6f} | PPL: {perplexity:8.4f} | "
                f"LR: {lr:.2e} | Grad: {grad_norm:.6f} | Epoch: {epoch:.3f}"
            )
            log_to_file(log_line)

# --- 3. LOAD MODEL & BASE ADAPTERS ---
log_to_file(f"⚙️ Loading Qwen2.5-VL Base Model in native bf16")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = torch.bfloat16, # pure precision
    load_in_4bit = False,   # Disable BitsAndBytes completely
    low_cpu_mem_usage = True,
)

log_to_file(f"🔧 Applying native Qwen2.5 ChatML template...")
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)

log_to_file(f"🧬 Injecting High-Fidelity DoRA Adapter...")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = False, # Protect her vision inference
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 64, # Massive cognitive density
    lora_alpha = 64,
    lora_dropout = 0.0,
    target_modules = "all-linear", # Target every node
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_dora = True,
    use_rslora = False,
    loftq_config = None,
)

# --- 4 & 5. DYNAMIC DATA FORMATTING & TRAINING LOOP ---
for stage in curriculum:
    log_to_file(f"\n🔥 STARTING STAGE: {stage['name'].upper()} (Target Context: {stage['max_seq']}, Optimizer: {stage['optim']})")

    output_dir = f"outputs_dora_{stage['name']}"
    resume_checkpoint = None

    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            resume_checkpoint = os.path.join(output_dir, checkpoints[-1])
            log_to_file(f"🔄 Found local checkpoint. Resuming from: {resume_checkpoint}")

    with open(stage['path'], 'r', encoding='utf-8') as f:
        raw_dataset = [json.loads(line) for line in f]

    dataset_system_prompt = ""
    for row in raw_dataset:
        if str(row.get("role", "")).lower() == "system":
            dataset_system_prompt = str(row.get("content", ""))
            break

    if dataset_system_prompt:
        dynamic_system_prompt = dataset_system_prompt
        log_to_file(f"🧠 Exclusively utilizing native system prompt from {stage['path']} shard.")
    else:
        log_to_file(f"⚠️ WARNING: No native system prompt found in {stage['path']}! System context will be blank.")
        dynamic_system_prompt = ""

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

            text = tokenizer.apply_chat_template(turn_messages, tokenize=False, add_generation_prompt=False)
            formatted_data.append({"text": text})

            current_conversation.append({"role": "assistant", "content": content})

        if row.get("reset_context") or len(current_conversation) > 12:
            current_conversation = []
            if dynamic_system_prompt:
                current_conversation.append({"role": "system", "content": dynamic_system_prompt})


    processed_dataset = Dataset.from_list(formatted_data)


    FastVisionModel.for_training(model)
    model.config.max_position_embeddings = stage['max_seq']

    trainer = SFTTrainer(
        model = model,
        train_dataset = processed_dataset,
        dataset_text_field = "text",
        tokenizer = tokenizer,
        max_seq_length = stage['max_seq'],
        dataset_num_proc = 2,
        packing = stage['pack'],
        callbacks = [FancyProgressCallback(stage['name'])],
        args = TrainingArguments( # 🔥 Reverted back to standard TrainingArguments
            num_train_epochs = stage['epochs'],
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8,
            gradient_checkpointing = True,
            gradient_checkpointing_kwargs = {"use_reentrant": False},
            warmup_steps = 10,
            learning_rate = stage['learning_rate'],
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            disable_tqdm = True,
            optim = stage['optim'],
            weight_decay = 0.01,
            output_dir = output_dir,
            save_strategy = "steps",
            save_steps = 50,
            save_total_limit = 5,
            load_best_model_at_end = False,
            average_tokens_across_devices = False,
        ),
    )

    trainer.remove_callback(PrinterCallback)

    log_to_file("Applying 'train_on_responses_only' mask to protect prompts from loss calculation.")
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part    = "<|im_start|>assistant\n"
    )

    # 🩹 THE UNSLOTH PICKLING HOTFIX
    # We forcefully un-mask the dynamic class so PyTorch can safely write it to the NVMe drive
    trainer.args.__class__ = TrainingArguments

    trainer.train(resume_from_checkpoint = resume_checkpoint)

    model.save_pretrained(f"dora_lora_after_{stage['name']}")
    log_to_file(f"✅ Stage {stage['name']} complete. Weights updated.")

    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    log_to_file(f"🧹 Slate cleaned. VRAM completely flushed.")

log_to_file("\n✨ MARATHON COMPLETE.")
