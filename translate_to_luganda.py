"""
MASSIVE Luganda Translator (Task 1.2)
=====================================
Uses Gemini to translate English utterances from MASSIVE into natural Luganda.
Output is used for Task 1.2 fine-tuning dataset.
"""

import os, json, time, random
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ─── Config ──────────────────────────────────────────────────────────────────

load_dotenv(override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = 'gemini-3.1-flash-lite-preview'

INPUT_PATH = Path("dataset/massive_converted/en-US_toolcalls.jsonl")
OUTPUT_PATH = Path("dataset/massive_converted/lg-UG_toolcalls.jsonl")
SAMPLE_SIZE = 200
BATCH_SIZE = 5  # processing in small batches

# ─── Translation Logic ───────────────────────────────────────────────────────

def translate_batch(utterances):
    """Translates a list of utterances using the new google-genai SDK."""
    results = []
    for utt in utterances:
        try:
            prompt = f"Translate this English utterance to natural Luganda: '{utt}'\n\nOutput ONLY the Luganda text."
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt
            )
            translated = response.text.strip()
            # Basic sanity check (strip quotes)
            translated = translated.strip('"').strip("'")
            results.append(translated)
            time.sleep(1) # Be gentle
        except Exception as e:
            print(f"  ⚠ Translation error for '{utt}': {e}")
            results.append(None)
    return results

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_PATH.exists():
        print(f"❌ Input file not found: {INPUT_PATH}")
        return

    # Load all EN examples
    with open(INPUT_PATH, encoding="utf-8") as f:
        all_exs = [json.loads(line) for line in f if line.strip()]

    # Sample 200 random ones
    selected = random.sample(all_exs, min(SAMPLE_SIZE, len(all_exs)))
    print(f"🚀 Translating {len(selected)} examples to Luganda...")

    lg_examples = []
    
    for i in range(0, len(selected), BATCH_SIZE):
        batch = selected[i:i+BATCH_SIZE]
        utts = [ex["user_prompt"] for ex in batch]
        
        print(f"  Processing batch {i//BATCH_SIZE + 1}/{(len(selected)-1)//BATCH_SIZE + 1}...", end=" ", flush=True)
        translations = translate_batch(utts)
        
        for ex, trans in zip(batch, translations):
            if trans:
                new_ex = ex.copy()
                new_ex["id"] = ex["id"].replace("en-US", "lg-UG")
                new_ex["language"] = "lg"
                new_ex["user_prompt"] = trans
                # Metadata remains same as it's the same intent
                lg_examples.append(new_ex)
        
        print(f"✓ ({len(lg_examples)} total Luganda examples so far)")

    # Save to file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in lg_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✅ DONE — {len(lg_examples)} Luganda examples saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
