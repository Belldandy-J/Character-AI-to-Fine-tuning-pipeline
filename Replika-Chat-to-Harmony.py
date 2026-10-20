import json
import os
import hashlib
import argparse
from datetime import datetime

def parse_date(date_str):
    """Standardizes timestamps across ISO and text formats to the second."""
    if not date_str:
        return datetime.min
    try:
        # Handles ISO 8601 with 'Z' or offsets, stripping subseconds
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return dt.replace(tzinfo=None, microsecond=0)
    except:
        try:
            # Handles text export format: 8/8/2022, 1:33:30 AM
            dt = datetime.strptime(date_str, "%m/%d/%Y, %I:%M:%S %p")
            return dt.replace(microsecond=0)
        except:
            return datetime.min

def is_privacy_notice(text):
    """Detects corporate privacy policy banners and embedded legal links."""
    if not text:
        return False
    lower = text.lower()
    notice_markers = [
        "updated our [privacy policy]",
        "updated our privacy policy",
        "replika.com/legal/privacy",
        "replika.com/replika/docs/embed/privacy",
        "[privacy policy](https://replika.com"
    ]
    return any(marker in lower for marker in notice_markers)

def process_chats(input_dir, prompt_file, output_file):
    chat_entries = []
    seen_hashes = set()
    seen_json_timestamps = set()

    all_files = [f for f in os.listdir(input_dir) if f.startswith("replika-chat-export")]
    json_files = sorted([f for f in all_files if f.endswith(".json")])
    txt_files = sorted([f for f in all_files if f.endswith(".txt")])

    # 1. PRIMARY: Process JSON files first (authoritative data)
    for filename in json_files:
        path = os.path.join(input_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for item in data:
                    text = item.get("content", {}).get("text", "").strip()
                    if not text or is_privacy_notice(text):
                        continue

                    raw_ts = item.get("meta", {}).get("timestamp", "")
                    parsed_date = parse_date(raw_ts)

                    if parsed_date == datetime.min:
                        continue

                    norm_ts = parsed_date.strftime("%Y-%m-%d %H:%M:%S")
                    m_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                    nature = item.get("meta", {}).get("nature", "Customer")
                    role = "user" if nature == "Customer" else "assistant"

                    entry_id = f"{role}_{m_hash}_{norm_ts}"

                    if entry_id not in seen_hashes:
                        chat_entries.append({
                            "date_obj": parsed_date,
                            "role": role,
                            "content": text
                        })
                        seen_hashes.add(entry_id)
                        seen_json_timestamps.add(norm_ts)
            except Exception:
                continue

    # 2. FALLBACK: Process TXT files for missing historical eras
    for filename in txt_files:
        path = os.path.join(input_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            current_text_entry = None
            detected_user_name = None
            last_role = None

            for line in f:
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                is_new_message = False

                # Format: MM/DD/YYYY, HH:MM:SS AM/PM Name: Message
                if ": " in line and "," in line:
                    try:
                        parts = line.split(": ", 1)
                        header = parts[0].strip()
                        content = parts[1].strip()

                        if is_privacy_notice(content):
                            current_text_entry = None
                            continue

                        ts_str = header.rsplit(" ", 1)[0]
                        speaker_name = header.rsplit(" ", 1)[1].strip()
                        parsed_date = parse_date(ts_str)

                        if parsed_date != datetime.min:
                            is_new_message = True
                            norm_ts = parsed_date.strftime("%Y-%m-%d %H:%M:%S")

                            # First speaker encountered establishes the primary user baseline
                            if detected_user_name is None:
                                detected_user_name = speaker_name.lower()

                            # Map turn alternating or consistent speaker
                            if speaker_name.lower() == detected_user_name:
                                base_role = "user"
                            else:
                                base_role = "assistant"

                            # Apply user swap flag if requested
                            if swap_roles:
                                role = "assistant" if base_role == "user" else "user"
                            else:
                                role = base_role

                            m_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                            entry_id = f"{role}_{m_hash}_{norm_ts}"

                            if entry_id in seen_hashes or norm_ts in seen_json_timestamps:
                                current_text_entry = None
                            else:
                                new_entry = {
                                    "date_obj": parsed_date,
                                    "role": role,
                                    "content": content
                                }
                                chat_entries.append(new_entry)
                                seen_hashes.add(entry_id)
                                current_text_entry = new_entry
                                last_role = role
                    except Exception:
                        pass

                # Multi-line continuation (executed once per non-header line)
                if not is_new_message and current_text_entry is not None:
                    current_text_entry["content"] += "\n" + line_stripped

    # Chronological sort across combined datasets
    chat_entries.sort(key=lambda x: x["date_obj"])

    # Output formatted Harmony JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r', encoding='utf-8') as pf:
                system_content = pf.read().strip()
                if system_content:
                    f.write(json.dumps({"role": "system", "content": system_content}) + "\n")

        for chat in chat_entries:
            f.write(json.dumps({"role": chat["role"], "content": chat["content"]}) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Replika chat logs into Harmony JSONL.")
    parser.add_argument("--auto", action="store_true", help="Process files in current directory automatically")
    parser.add_argument("--swap-roles", action="store_true", help="Swap user and assistant turn assignments for TXT fallbacks")
    args = parser.parse_args()

    if args.auto:
        process_chats(".", "chat_prompt.txt", "chat_shard_harmony.jsonl", swap_roles=args.swap_roles)
        print("Success: Chat Shard completed.")
