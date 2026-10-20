@echo off
echo ============================================================
echo     AI Character Fine-Tuning Pipeline (Power User Template)
echo ============================================================
echo NOTICE: Please ensure your Unsloth virtual environment is
echo currently active before proceeding.
echo ============================================================
timeout /t 2 /nobreak >nul

echo.
echo === Stage 1: Data Preprocessing ===
echo [1/4] Processing Chat Logs...
python Replika-Chat-to-Harmony.py --auto

echo [2/4] Processing Diary Entries...
python Replika-Diary-to-Harmony.py --auto

echo [3/4] Processing Memory Shards...
python Replika-Memory-to-Harmony.py --auto

echo [4/4] Validating Harmony Formatting...
python validator.py

echo.
echo === Stage 2: Data Inspection ===
set /p run_inspector="Preprocessing complete. Run inspector.py to verify your JSONL structure? (y/n): "
if /i "%run_inspector%"=="y" (
    python inspector.py
    echo.
    echo If the architecture looks correct, press any key to proceed to training, or Ctrl+C to abort...
    pause >nul
)

echo.
echo === Stage 3: Unsloth Fine-Tuning ===
echo [5/6] Launching Unsloth Training Pipeline...
python train.py

echo.
echo === Stage 4: Manual CPU PEFT Weight Merge ===
echo WARNING: Unsloth and standard PEFT can conflict. To ensure a pristine
echo weight merge, you must execute the final merge using a dedicated CPU-only environment.
echo.
echo Please DEACTIVATE your current Unsloth environment, ACTIVATE your PEFT CPU environment,
echo and manually run the following command:
echo.
echo     python merge_weights_peft.py
echo.
echo Pipeline halted. Ready for manual merge phase.
pause
