"""
Tiny Facade — Dataset Validator (Task 1.2)
==========================================
Loads the generated JSONL dataset and prints stats + random samples.

Usage:
  python validate_dataset.py
  python validate_dataset.py --file dataset/tool_calling_dataset.jsonl
"""

import json, random, argparse
from pathlib import Path
from collections import defaultdict

def validate(path: str):
    p = Path(path)
    if not p.exists():
        print(f"❌ File not found: {path}")
        return

    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n📂 Dataset: {path}")
    print(f"   Total lines: {len(lines)}")

    examples = []
    parse_errors = 0
    for line in lines:
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError:
            parse_errors += 1

    print(f"   Parse errors: {parse_errors}")
    print(f"   Valid examples: {len(examples)}\n")

    # Per-language counts
    by_lang = defaultdict(list)
    by_tool = defaultdict(int)
    issues = []

    for ex in examples:
        lang = ex.get("language", "?")
        tool = ex.get("tool_name", "?")
        by_lang[lang].append(ex)
        by_tool[tool] += 1

        # Validate structure
        if not ex.get("user_prompt"):
            issues.append(f"  [{ex.get('id')}] Missing user_prompt")
        out = ex.get("expected_output", {})
        if not isinstance(out, dict):
            issues.append(f"  [{ex.get('id')}] expected_output not a dict")
        elif out.get("name") != tool:
            issues.append(f"  [{ex.get('id')}] name mismatch: {out.get('name')} vs {tool}")
        elif not isinstance(out.get("arguments"), dict):
            issues.append(f"  [{ex.get('id')}] arguments not a dict")
        elif not out.get("arguments"):
            issues.append(f"  [{ex.get('id')}] arguments is empty")

    # Language summary
    print("─" * 50)
    print(f"{'Language':<12} {'Count':>6}  {'Target':>6}  {'%':>6}")
    print("─" * 50)
    lang_map = {"en": "English", "sw": "Swahili", "lg": "Luganda"}
    for code in ["en", "sw", "lg"]:
        count = len(by_lang.get(code, []))
        pct = f"{count/200*100:.0f}%" if count > 0 else "0%"
        print(f"  {lang_map.get(code, code):<10} {count:>6}  {200:>6}  {pct:>6}")
    print("─" * 50)
    print(f"  {'TOTAL':<10} {len(examples):>6}  {600:>6}  {len(examples)/600*100:.0f}%\n")

    # Tool coverage
    print("─" * 50)
    print("Tool coverage:")
    print("─" * 50)
    for tool, count in sorted(by_tool.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 3)
        print(f"  {tool:<22} {count:>4}  {bar}")
    print()

    # Issues
    if issues:
        print(f"⚠  {len(issues)} structural issue(s) found:")
        for iss in issues[:20]:
            print(iss)
        if len(issues) > 20:
            print(f"  … and {len(issues) - 20} more.")
    else:
        print("✅ No structural issues found.\n")

    # Random samples per language
    print("─" * 50)
    print("Random samples (3 per language):")
    print("─" * 50)
    for code in ["en", "sw", "lg"]:
        lang_exs = by_lang.get(code, [])
        samples = random.sample(lang_exs, min(3, len(lang_exs)))
        print(f"\n  [{lang_map.get(code, code).upper()}]")
        for s in samples:
            print(f"    id: {s.get('id')}")
            print(f"    user: {s.get('user_prompt')}")
            print(f"    call: {json.dumps(s.get('expected_output'), ensure_ascii=False)}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Tiny Facade Dataset Validator")
    parser.add_argument("--file", default="dataset/tool_calling_dataset.jsonl", help="Path to JSONL dataset file")
    args = parser.parse_args()
    validate(args.file)


if __name__ == "__main__":
    main()
