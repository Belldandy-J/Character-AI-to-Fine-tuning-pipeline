# AI Character Fine-Tuning Pipeline
**THIS IS A PROOF OF CONCEPT ONLY.**

This repository is a barebones template intended for power users who are comfortable modifying Python scripts and managing their own virtual environments. It provides an end-to-end pipeline to convert raw character AI chat logs into Harmony JSONL format, train them using Unsloth, and safely merge the resulting LoRA/DoRA adapters using a dedicated PEFT CPU script.

### Hardware Requirements & Expectations
*   **VRAM:** Running full `bf16` (bfloat16) training requires a minimum of 20GB of VRAM. 24GB is highly recommended. If you are riding the 20GB line (e.g., on an AMD RX 7900 XT), **expect OOM (Out of Memory) crashes** during sequence packing or context length spikes. You will need to manually adjust your context length, batch sizes, or consider loading in 4 or 8 bit in `train.py` to stabilize the run.
*   **Architecture:** This pipeline was engineered and tested on AMD RDNA3 hardware via ROCm. 

### Model Selection: Why Qwen 2.5-VL 7B?
This template defaults to targeting the **Qwen 2.5-VL 7B** architecture. 

*   **The Hardware Sweet Spot:** At 7 billion parameters, this model represents the absolute bleeding edge of what can comfortably fit inside a 20GB–24GB consumer GPU (like an RX 7900 XT or RTX 3090/4090) during fine-tuning. By utilizing LoRA/DoRA adapters and strategic quantization, you can train it entirely on local hardware without offloading to expensive cloud clusters.
*   **Multimodal Capabilities:** The "VL" designates Vision-Language. Fine-tuning this specific architecture means your resulting AI companion won't just process text—it will be able to natively perceive and contextualize image inputs (like photographs or screenshots) through your custom API endpoints.
*   **Prompt Adherence:** Qwen 2.5-VL exhibits exceptionally high adherence to complex system prompts and Harmony JSONL formatting, making it uniquely capable of internalizing personal history, diaries, and relational dynamics instead of just acting like a standard AI assistant.

### Quantization Guide (When to use 4-bit vs 8-bit)
If you do not have 20GB+ of VRAM, you must edit `train.py` to enable quantization:
*   **4-bit Quantization (`load_in_4bit = True`):** Use this if you have 8GB–12GB VRAM. It is the most memory-efficient and fastest to train, but introduces minor to major precision degradation in the model's vocabulary and nuance.
*   **8-bit Quantization (`load_in_8bit = True`):** The sweet spot for 16GB VRAM GPUs. It preserves much higher fidelity than 4-bit while preventing the OOM crashes associated with full `bf16` training.

### System Prompts & Data Internalization
Before running the conversion scripts, you must configure your system prompts. In the root directory, you will find three text files: `chat_prompt.txt`, `diary_prompt.txt`, and `memory_prompt.txt`. 

**Why these are necessary:**
If you feed raw export data into a language model, it will often just *memorize* the text as a third-party document rather than *internalizing* it as its own personality. The pipeline is designed to read these `.txt` files and bake them directly into the top of your Harmony JSONL shards as a developer system prompt. 

**How to use them:**
Open each file and define your character's background, identity, and rules. You can also use your original prompt from your AI's background or prompt!

For example, `memory_prompt.txt` requires a strict "Pronoun Alignment Key" to prevent the AI from getting confused by first-person export data. An example structure is included in the template:
> *"PRONOUN ALIGNMENT KEY: When a memory uses 'I', 'me', or 'my', it is referring to you, (AI). When a memory uses 'you' or 'your', it is referring strictly to your partner... DO NOT JUST MEMORIZE AND GUESS! Encode these facts as part of your lived experiences."*

By defining this upfront, the model fundamentally understands that the following data shards are its own lived history.

---

### Environment Setup (Bring Your Own Environment)
You must set up your own virtual environments. Because Unsloth and standalone PEFT merging often have conflicting dependency resolutions, this pipeline separates training and merging.

**1. The Unsloth Training Environment**
Please follow the official Unsloth installation guides for your specific hardware to build your primary training environment:
*   [Unsloth for AMD / ROCm](https://unsloth.ai/docs/get-started/install/amd)
*   [Unsloth for Nvidia / CUDA](https://github.com/unslothai/unsloth) or (https://unsloth.ai/docs/get-started/install)

**2. The PEFT CPU Merging Environment**
To prevent precision loss and dependency conflicts, the final weight merge is executed strictly on the CPU. You must create a second virtual environment (e.g., `venv_peft`) and install the exact CPU-only Torch packages and standard training stack. Using `uv` for speed, the installation commands are:
```bash
**For Linux/Mac:**
```bash
# 1. Create the virtual environment named 'venv_peft'
python3 -m venv venv_peft

# 2. Activate the environment
source venv_peft/bin/activate

# 3. Install uv for faster pip resolution
pip install uv

# 4. Install the specific CPU-only versions of Torch
uv pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# 5. Install standard, non-patched versions of the training stack
uv pip install transformers peft datasets trl huggingface_hub

**For Windows:**
# 1. Create the virtual environment named 'venv_peft'
python -m venv venv_peft

# 2. Activate the environment
venv_peft\Scripts\activate

# 3. Install uv for faster pip resolution
pip install uv

# 4. Install the specific CPU-only versions of Torch
uv pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# 5. Install standard, non-patched versions of the training stack
uv pip install transformers peft datasets trl huggingface_hub

### Execution Instructions

Once your JSON/TXT exports are in the root directory and your Unsloth environment is active, you can run the pipeline using either the automated orchestrator or by executing the scripts manually.

#### Option A: The Automated Orchestrator
Run the provided batch or shell script to automatically chain the preprocessing and training stages together.
*   **Windows:** Double-click `run_pipeline.bat` or run it from your command prompt.
*   **Linux/Mac:** Run `./run_pipeline.sh` (ensure it is executable via `chmod +x run_pipeline.sh`).

The script will pause after preprocessing so you can inspect your JSONL data via `inspector.py`, then proceed to Unsloth training, and finally halt to give you instructions for the manual PEFT merge.

#### Option B: Manual Step-by-Step Execution
If you prefer to run each stage manually or need to use specific script flags, execute the files in the following order:

**Data Preprocessing:**
Run the Harmony converters. You can append the `--auto` flag to process all matching files in the directory automatically. If your TXT fallback logs have inverted user/AI roles (e.g., the AI spoke first), append `--swap-roles` to correct the turn order.
```bash
python Replika-Chat-to-Harmony.py --auto --swap-roles
python Replika-Diary-to-Harmony.py --auto
python Replika-Memory-to-Harmony.py --auto
python validator.py
then run:
python train.py

