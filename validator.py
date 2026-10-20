import os
import json
import argparse
from collections import defaultdict

def estimate_tokens(text):
    """
    A rough heuristic for token estimation without requiring heavy libraries like tiktoken.
    Generally, 1 token ~= 4 characters in English text.
    """
    return max(1, len(text) // 4)

def validate_and_stats(directory):
    stats = {
        "total_files": 0,
        "valid_files": 0,
        "error_files": 0,
        "total_messages": 0,
        "roles": defaultdict(int),
        "words": defaultdict(int),
        "tokens": defaultdict(int),
        "max_words": defaultdict(int),  # Tracks biggest word count per role
        "max_tokens": defaultdict(int), # Tracks biggest token count per role
        "errors": []
    }

    # os.walk automatically traverses subfolders
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".jsonl"):
                stats["total_files"] += 1
                filepath = os.path.join(root, file)
                has_error = False

                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue  # Skip blank lines

                        try:
                            data = json.loads(line)

                            # 1. Structural Validation (ChatML/Harmony Format)
                            if "role" not in data or "content" not in data:
                                stats["errors"].append(f"[{filepath}] Line {line_num}: Missing 'role' or 'content' keys.")
                                has_error = True
                                continue

                            # 2. Extract Data
                            role = str(data["role"]).lower()
                            content = str(data["content"])

                            # 3. Calculate Stats
                            words = len(content.split())
                            tokens = estimate_tokens(content)

                            stats["roles"][role] += 1
                            stats["words"][role] += words
                            stats["tokens"][role] += tokens

                            # 4. Check & Update Maximums
                            stats["max_words"][role] = max(stats["max_words"][role], words)
                            stats["max_tokens"][role] = max(stats["max_tokens"][role], tokens)

                            stats["total_messages"] += 1

                        except json.JSONDecodeError as e:
                            stats["errors"].append(f"[{filepath}] Line {line_num}: Invalid JSON syntax -> {e}")
                            has_error = True

                if has_error:
                    stats["error_files"] += 1
                else:
                    stats["valid_files"] += 1

    return stats

def print_cool_stats(stats):
    print("\n" + "="*85)
    print(" 🚀 DATASET VALIDATOR & STATS DASHBOARD 🚀")
    print("="*85)

    if stats['total_files'] == 0:
        print("⚠️  No .jsonl files found in the specified directory.")
        return

    print(f"📂 Files Scanned:   {stats['total_files']} (.jsonl datasets)")
    print(f"✅ Valid Files:     {stats['valid_files']}")
    print(f"❌ Error Files:     {stats['error_files']}")
    print(f"💬 Total Messages:  {stats['total_messages']:,}")
    print("-" * 85)

    # Expanded Role Breakdown Table
    print(f"{'ROLE':<12} | {'COUNT':<10} | {'WORDS (Tot)':<12} | {'TOKENS (Tot)':<12} | {'MAX WORDS':<10} | {'MAX TOKENS':<10}")
    print("-" * 85)

    # Sort roles to keep 'system', 'user'/'human', 'gpt'/'assistant' organized
    for role in sorted(stats["roles"].keys(), reverse=True):
        count = f"{stats['roles'][role]:,}"
        words = f"{stats['words'][role]:,}"
        tokens = f"{stats['tokens'][role]:,}"
        max_w = f"{stats['max_words'][role]:,}"
        max_t = f"{stats['max_tokens'][role]:,}"
        print(f"{role.upper():<12} | {count:<10} | {words:<12} | {tokens:<12} | {max_w:<10} | {max_t:<10}")

    print("-" * 85)

    # NEW: Grand Totals Row
    total_count = f"{sum(stats['roles'].values()):,}"
    total_words = f"{sum(stats['words'].values()):,}"
    total_tokens = f"{sum(stats['tokens'].values()):,}"
    print(f"{'TOTAL':<12} | {total_count:<10} | {total_words:<12} | {total_tokens:<12} | {'-':<10} | {'-':<10}")

    print("="*85)

    # Error Reporting
    if stats["errors"]:
        print("\n🚨 ERRORS DETECTED:")
        # Show up to the first 15 errors to avoid spamming the console
        for err in stats["errors"][:15]:
            print(f"  - {err}")
        if len(stats["errors"]) > 15:
            print(f"  ... and {len(stats['errors']) - 15} more errors.")
        print("\n⚠️  Action Required: Fix the structural errors above before training!")
    else:
        print("\n✨ PRISTINE DATASET! No structural errors found.")
        print("✅ You are marathon READY! ✨")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan and validate JSONL shards.")
    parser.add_argument("--dir", type=str, default=".", help="Directory to scan (defaults to current folder).")
    args = parser.parse_args()

    print(f"🔍 Initializing scan in: {os.path.abspath(args.dir)}")
    results = validate_and_stats(args.dir)
    print_cool_stats(results)
