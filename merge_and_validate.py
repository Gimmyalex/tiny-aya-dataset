"""
Final Dataset Merger and Validator (Task 1.2)
============================================
Combines EN, SW, and LG tool-calling examples into the final training set.
Performs structural validation and balance checks.
"""

import json, os, random
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

SOURCE_DIR = Path("dataset/massive_converted")
OUTPUT_FILE = Path("dataset/tool_calling_dataset.jsonl")

LANGUAGE_FILES = {
    "en": SOURCE_DIR / "en-US_toolcalls.jsonl",
    "sw": SOURCE_DIR / "sw-KE_toolcalls.jsonl",
    "lg": SOURCE_DIR / "lg-UG_toolcalls.jsonl"
}

SAMPLE_PER_LANG = 200

def validate_example(ex):
    """Checks for required fields and basic tool call structure."""
    required = ["id", "language", "user_prompt", "expected_output"]
    for r in required:
        if r not in ex: return False, f"Missing {r}"
    
    out = ex["expected_output"]
    if "name" not in out or "arguments" not in out:
        return False, "Malformed expected_output"
    
    return True, ""

def main():
    final_data = []
    stats = {lang: {"total": 0, "tools": {}} for lang in LANGUAGE_FILES}

    print("🧩 Merging datasets...")

    for lang, path in LANGUAGE_FILES.items():
        if not path.exists():
            print(f"  ⚠ File not found for {lang}: {path}")
            continue
            
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
            
            # For EN/SW we have thousands, so sample 200
            # For LG we take all (since we already sampled 200 for translation)
            if len(lines) > SAMPLE_PER_LANG:
                selected = random.sample(lines, SAMPLE_PER_LANG)
            else:
                selected = lines
            
            for ex in selected:
                valid, err = validate_example(ex)
                if valid:
                    final_data.append(ex)
                    stats[lang]["total"] += 1
                    tname = ex["tool_name"]
                    stats[lang]["tools"][tname] = stats[lang]["tools"].get(tname, 0) + 1
                else:
                    print(f"  ⚠ Invalid {lang} example {ex.get('id')}: {err}")

    # Shuffle everything for better training balance
    random.shuffle(final_data)

    # Save final
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in final_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✅ Merged {len(final_data)} examples into {OUTPUT_FILE}")
    print("\n📈 Dataset Statistics:")
    for lang, data in stats.items():
        print(f"  {lang.upper()}: {data['total']} examples")
        # Print top tools
        top_tools = sorted(data["tools"].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"    Top tools: {', '.join([f'{t} ({c})' for t, c in top_tools])}")

    print("\n👀 Samples (one per language):")
    for lang in LANGUAGE_FILES:
        sample = next((ex for ex in final_data if ex["language"] == lang), None)
        if sample:
            print(f"  [{lang.upper()}] Prompt: {sample['user_prompt']}")
            print(f"       Call: {json.dumps(sample['expected_output'], ensure_ascii=False)}")

if __name__ == "__main__":
    main()
