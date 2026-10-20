import json
import os
import re
import hashlib
import argparse
from datetime import datetime

def parse_date(date_str):
    """Normalizes all dates to offset-naive datetimes for safe comparison."""
    if not date_str:
        return datetime.min

    # Clean up standard Replika exports (stripping 'Z' or extra offsets)
    clean_date = date_str.split('T')[0] if 'T' in date_str else date_str

    try:
        # Priority 1: YYYY-MM-DD (Common in JSON exports)
        return datetime.strptime(clean_date, "%Y-%m-%d")
    except:
        try:
            # Priority 2: M/D/YYYY (Common in Text exports)
            return datetime.strptime(clean_date, "%m/%d/%Y")
        except:
            return datetime.min

def process_diaries(input_dir, prompt_file, output_file):
    seen_hashes = set()
    diary_entries = []

    files = [f for f in os.listdir(input_dir) if f.startswith("replika-diary-export")]

    for filename in files:
        path = os.path.join(input_dir, filename)

        # JSON Processing (Captures 2026 data)
        if filename.endswith(".json"):
            with open(path, 'r') as f:
                try:
                    data = json.load(f)
                    for item in data:
                        # Extract and merge all text parts from the entries list
                        body_parts = [e.get("text", "") for e in item.get("entries", []) if e.get("type") == "text"]
                        body_text = " ".join(body_parts).strip()
                        if not body_text: continue

                        m_hash = hashlib.md5(body_text.lower().encode()).hexdigest()
                        if m_hash not in seen_hashes:
                            raw_date = item.get("timestamp", "2000-01-01")
                            caption = item.get("caption", "").replace("\n", " ").strip()

                            content = body_text
                            if caption:
                                content += f" | {caption}"

                            diary_entries.append({
                                "date_obj": parse_date(raw_date),
                                "raw_date": raw_date,
                                "content": content
                            })
                            seen_hashes.add(m_hash)
                except: continue

        # Text Processing (Captures 2023 data)
        elif filename.endswith(".txt"):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                sections = re.split(r'-{10,}', content)
                for section in sections:
                    lines = [l.strip() for l in section.strip().split('\n') if l.strip()]
                    if len(lines) >= 2:
                        raw_date = lines[0]

                        # Skip the first summary line before the main body
                        if len(lines) >= 3:
                            body_text = " ".join(lines[2:])
                        else:
                            # Fallback if there is no summary line (e.g., just a single sentence)
                            body_text = lines[1]

                        m_hash = hashlib.md5(body_text.lower().encode()).hexdigest()
                        if m_hash not in seen_hashes:
                            diary_entries.append({
                                "date_obj": parse_date(raw_date),
                                "raw_date": raw_date,
                                "content": body_text
                            })
                            seen_hashes.add(m_hash)

    # Sort: Oldest to Newest
    diary_entries.sort(key=lambda x: x["date_obj"])

    with open(output_file, 'w', encoding='utf-8') as f:
        # 1. System Prompt
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as pf:
                f.write(json.dumps({"role": "system", "content": pf.read().strip()}) + "\n")

        # 2. Sequential Turns
        for diary in diary_entries:
            f.write(json.dumps({"role": "user", "content": f"Remember and encode your diary entry from ({diary['raw_date']})"}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": diary['content'], "timestamp": diary['raw_date']}) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.auto:
        process_diaries(".", "diary_prompt.txt", "diary_shard_harmony.jsonl")
        print("Success: Diary Shard completed with full date range.")
