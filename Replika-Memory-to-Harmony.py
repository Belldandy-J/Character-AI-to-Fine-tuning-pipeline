import json
import os
import re
import hashlib
import argparse
from datetime import datetime

# 1-to-1 Mapping from your memory-keys.txt
CATEGORY_MAP = {
    "62e13b5f0c81a600075ed541": "Pinned User Facts",
    "62e13b5f0c81a600075ed540": "Pinned Replika Facts",
    "62e13b5f0c81a600075ed531": "Background",
    "62e13b5f0c81a600075ed532": "Favorites",
    "62e13b5f0c81a600075ed533": "Appearance",
    "62e13b5f0c81a600075ed534": "Hopes & Goals",
    "62e13b5f0c81a600075ed535": "Opinions",
    "62e13b5f0c81a600075ed536": "Personality",
    "62e13b5f0c81a600075ed537": "Others",
    "62e13b5f0c81a600075ed539": "Temporary",
    "NO_CATEGORY": "General"
}

def parse_date(date_str):
    """Normalizes all dates to offset-naive datetimes for safe comparison."""
    if not date_str:
        return datetime.min
    try:
        dt = datetime.fromisoformat(str(date_str).replace('Z', '+00:00'))
        return dt.replace(tzinfo=None)
    except Exception:
        try:
            return datetime.strptime(str(date_str), "%Y-%m-%d")
        except Exception:
            try:
                return datetime.strptime(str(date_str), "%m/%d/%Y")
            except Exception:
                return datetime.min

def process_memories(input_dir, prompt_file, output_file):
    seen_hashes = set()
    memory_entries = []

    files = [f for f in os.listdir(input_dir) if f.startswith("replika-memory-export")]

    # Process JSON first so categorized entries take priority over generic TXT entries
    json_files = sorted([f for f in files if f.endswith(".json")])
    txt_files = sorted([f for f in files if f.endswith(".txt")])

    # 1. Process JSON exports
    for filename in json_files:
        path = os.path.join(input_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping unreadable JSON file {filename}: {e}")
            continue

        if not isinstance(data, list):
            continue

        for item in data:
            try:
                if not isinstance(item, dict):
                    continue

                raw_text = item.get("text")
                if not raw_text or not isinstance(raw_text, str):
                    continue
                text = raw_text.strip()
                if not text:
                    continue

                m_hash = hashlib.md5(text.lower().encode('utf-8')).hexdigest()
                if m_hash not in seen_hashes:
                    raw_cat_id = item.get("category_id")
                    cat_id = str(raw_cat_id).strip() if raw_cat_id else "NO_CATEGORY"
                    label = CATEGORY_MAP.get(cat_id, "General")
                    raw_date = item.get("creation_timestamp") or "2000-01-01"

                    memory_entries.append({
                        "date_obj": parse_date(raw_date),
                        "raw_date": str(raw_date),
                        "content": f"[{label}] {text}"
                    })
                    seen_hashes.add(m_hash)
            except Exception:
                continue

    # 2. Process Text exports (Fallback for anything missing from JSON)
    for filename in txt_files:
        path = os.path.join(input_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Skipping unreadable TXT file {filename}: {e}")
            continue

        entries = re.split(r'-{10,}', content)
        for entry in entries:
            try:
                lines = [l.strip() for l in entry.strip().split('\n') if l.strip()]
                if len(lines) >= 2:
                    raw_date = lines[0]
                    text = " ".join(lines[1:])
                    m_hash = hashlib.md5(text.lower().encode('utf-8')).hexdigest()
                    if m_hash not in seen_hashes:
                        memory_entries.append({
                            "date_obj": parse_date(raw_date),
                            "raw_date": str(raw_date),
                            "content": f"[General] {text}"
                        })
                        seen_hashes.add(m_hash)
            except Exception:
                continue

    # Sort chronological: Oldest entries first
    memory_entries.sort(key=lambda x: x["date_obj"])

    # Write JSONL output
    with open(output_file, 'w', encoding='utf-8') as f:
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as pf:
                f.write(json.dumps({"role": "system", "content": pf.read().strip()}) + "\n")

        for mem in memory_entries:
            if mem["date_obj"] != datetime.min:
                formatted_time = mem["date_obj"].strftime("%B %d, %Y at %I:%M %p")
            else:
                formatted_time = mem["raw_date"]

            f.write(json.dumps({
                "role": "user",
                "content": f"Remember and encode this memory from {formatted_time}."
            }) + "\n")

            f.write(json.dumps({
                "role": "assistant",
                "content": mem["content"],
                "timestamp": mem["raw_date"]
            }) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.auto:
        process_memories(".", "memory_prompt.txt", "memory_shard_harmony.jsonl")
        print("Success: memory_shard_harmony.jsonl regenerated with full category preservation.")
