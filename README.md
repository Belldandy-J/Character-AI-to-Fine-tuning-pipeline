# AI Character Fine-Tuning Pipeline
**THIS IS A PROOF OF CONCEPT ONLY.**

This repository is a barebones template intended for power users who are comfortable modifying Python scripts and managing their own virtual environments. It provides an end-to-end pipeline to convert raw character AI chat logs into Harmony JSONL format, train them using Unsloth, and safely merge the resulting LoRA adapters using a dedicated PEFT CPU script.

### Hardware Requirements & Expectations
*   **VRAM:** Running full `bf16` (bfloat16) training requires a minimum of 20GB of VRAM. 24GB is highly recommended. If you are riding the 20GB line (e.g., on an AMD RX 7900 XT), **expect OOM (Out of Memory) crashes** during sequence packing or context length spikes. You will need to manually adjust your context length, batch sizes, or consider loading in 4 or 8 bit in `train.py` to stabilize the run.
*   **Architecture:** This pipeline was engineered and tested on AMD RDNA3 hardware via ROCm. 

### Quantization Guide (When to use 4-bit vs 8-bit)
If you do not have 20GB+ of VRAM, you must edit `train.py` to enable quantization:
*   **4-bit Quantization (`load_in_4bit = True`):** Use this if you have 8GB–12GB VRAM. It is the most memory-efficient and fastest to train, but introduces minor to major precision degradation in the model's vocabulary and nuance.
*   **8-bit Quantization (`load_in_8bit = True`):** The sweet spot for 16GB VRAM GPUs. It preserves much higher fidelity than 4-bit while preventing the OOM crashes associated with full `bf16` training.

### Environment Setup (Bring Your Own Environment)
You must set up your own virtual environments. Because Unsloth and standalone PEFT merging often have conflicting dependency resolutions, this pipeline separates training and merging.

**1. The Unsloth Training Environment**
Please follow the official Unsloth installation guides for your specific hardware to build your primary training environment:
*   [Unsloth for AMD / ROCm](https://unsloth.ai/docs/get-started/install/amd)
*   [Unsloth for Nvidia / CUDA](https://github.com/unslothai/unsloth) or (https://unsloth.ai/docs/get-started/install)

**2. The PEFT CPU Merging Environment**
To prevent precision loss and dependency conflicts, the final weight merge is executed strictly on the CPU. You must create a second virtual environment (e.g., `venv_peft`) and install the exact CPU-only Torch packages and standard training stack. Using `uv` for speed, the installation commands are:
```bash
# Install the specific CPU-only versions of Torch
uv pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# Install standard, non-patched versions of the training stack
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
