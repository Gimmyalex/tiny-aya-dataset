"""
Tiny Facade — Synthetic Dataset Generator (Task 1.2)
======================================================
Generates ~600 tool-calling examples across English (EN), Swahili (SW),
and Luganda (LG) using Cohere Command A.

Usage:
  python generate_dataset.py              # full run (600 examples)
  python generate_dataset.py --test       # test run (3 examples, 1 per language)
  python generate_dataset.py --lang en    # generate English only
  python generate_dataset.py --resume     # skip already-completed batches

Output:
  dataset/tool_calling_dataset.jsonl      # merged final dataset
  dataset/raw/<lang>_<tool>.jsonl         # per-batch intermediate files
"""

import os, json, time, random, argparse, re
from pathlib import Path
from datetime import datetime

import cohere
from dotenv import load_dotenv

# ─── Config ──────────────────────────────────────────────────────────────────

load_dotenv(override=True)  # override=True ensures .env wins over OS-level env vars

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    COHERE_API_KEY = input("Enter your Cohere API key: ").strip()

MODEL = "command-a-03-2025"
EXAMPLES_PER_LANG = 200        # 200 × 3 = 600 total
TOOLS_COUNT = 10
EXAMPLES_PER_BATCH = EXAMPLES_PER_LANG // TOOLS_COUNT  # 20 per (lang, tool) pair
DISTRACTOR_TOOLS = 3           # number of extra tool defs shown alongside the correct one
RETRY_LIMIT = 4
RETRY_DELAY = 5                # seconds (doubles on each retry)

DATASET_DIR = Path("dataset")
RAW_DIR = DATASET_DIR / "raw"
FINAL_PATH = DATASET_DIR / "tool_calling_dataset.jsonl"
RAW_DIR.mkdir(parents=True, exist_ok=True)

co = cohere.ClientV2(api_key=COHERE_API_KEY)

# ─── Tool Schemas ─────────────────────────────────────────────────────────────

TOOLS = {
    "get_weather": {
        "name": "get_weather",
        "description": "Get current weather conditions for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City or region name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
            },
            "required": ["location"]
        }
    },
    "search_web": {
        "name": "search_web",
        "description": "Search the web for information on a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"}
            },
            "required": ["query"]
        }
    },
    "calculate": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression to evaluate, e.g. '15 * 4 + 10'"}
            },
            "required": ["expression"]
        }
    },
    "translate_text": {
        "name": "translate_text",
        "description": "Translate text between languages.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "source_lang": {"type": "string", "description": "Source language code (e.g. 'sw', 'lg', 'en')"},
                "target_lang": {"type": "string", "description": "Target language code (e.g. 'en', 'sw', 'lg')"}
            },
            "required": ["text", "target_lang"]
        }
    },
    "set_reminder": {
        "name": "set_reminder",
        "description": "Set a reminder or alarm for a specific time.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Reminder message or label"},
                "datetime": {"type": "string", "description": "ISO 8601 datetime string, e.g. '2026-03-12T09:00:00'"}
            },
            "required": ["message", "datetime"]
        }
    },
    "get_news": {
        "name": "get_news",
        "description": "Fetch latest news headlines on a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "News topic or keyword"},
                "language": {"type": "string", "description": "Language code for news results"}
            },
            "required": ["topic"]
        }
    },
    "convert_currency": {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency code, e.g. 'UGX'"},
                "to_currency": {"type": "string", "description": "Target currency code, e.g. 'USD'"}
            },
            "required": ["amount", "from_currency", "to_currency"]
        }
    },
    "send_message": {
        "name": "send_message",
        "description": "Send a text message to a contact.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Name or phone number of recipient"},
                "message": {"type": "string", "description": "Message body text"}
            },
            "required": ["recipient", "message"]
        }
    },
    "create_event": {
        "name": "create_event",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "datetime": {"type": "string", "description": "ISO 8601 datetime for the event"},
                "location": {"type": "string", "description": "Event location (optional)"}
            },
            "required": ["title", "datetime"]
        }
    },
    "get_directions": {
        "name": "get_directions",
        "description": "Get directions between two locations.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Starting location"},
                "destination": {"type": "string", "description": "Destination location"},
                "mode": {"type": "string", "enum": ["driving", "walking", "transit"], "description": "Travel mode"}
            },
            "required": ["origin", "destination"]
        }
    }
}

TOOL_NAMES = list(TOOLS.keys())

# ─── Language Config ──────────────────────────────────────────────────────────

LANG_CONFIG = {
    "en": {
        "name": "English",
        "instruction": "in natural, everyday English",
        "context": "Users are from East Africa (Uganda, Kenya, Tanzania) but speaking English.",
        "example_locations": ["Kampala", "Nairobi", "Dar es Salaam", "Entebbe", "Mombasa"],
        "example_names": ["Sarah", "James", "Mama", "Baba", "Grace", "John", "Esther"]
    },
    "sw": {
        "name": "Swahili",
        "instruction": "in natural, everyday Swahili (Kiswahili). The user utterance must be in Swahili. All arguments in the JSON output should be in English.",
        "context": "Users are East African Swahili speakers from Tanzania, Kenya, or Uganda.",
        "example_locations": ["Kampala", "Nairobi", "Dar es Salaam", "Mombasa", "Zanzibar"],
        "example_names": ["Mama", "Baba", "Amina", "Hassan", "Fatuma", "Juma", "Zawadi"]
    },
    "lg": {
        "name": "Luganda",
        "instruction": "in natural, everyday Luganda. The user utterance must be in Luganda. All arguments in the JSON output should be in English.",
        "context": "Users are Baganda speakers from Uganda, primarily around Kampala and Central Uganda.",
        "example_locations": ["Kampala", "Entebbe", "Mukono", "Wakiso", "Masaka", "Jinja"],
        "example_names": ["Mama", "Taata", "Nakato", "Babirye", "Kato", "Wasswa", "Nambi", "Tendo"]
    }
}

# ─── Prompt Builder ───────────────────────────────────────────────────────────

def build_generation_prompt(lang_code: str, tool_name: str, n: int) -> str:
    lang = LANG_CONFIG[lang_code]
    tool = TOOLS[tool_name]
    tool_schema_str = json.dumps(tool, indent=2, ensure_ascii=False)

    return f"""You are a dataset generator for a multilingual AI training project called Tiny Facade.
Your task: generate {n} diverse, realistic tool-calling training examples {lang["instruction"]}.

TOOL TO TARGET: {tool_name}
{lang["context"]}
Typical locations: {', '.join(lang["example_locations"])}
Typical names: {', '.join(lang["example_names"])}

TARGET TOOL SCHEMA:
{tool_schema_str}

RULES:
1. Each user utterance must be a natural, realistic request {lang["instruction"]}.
2. The user utterance should naturally require the {tool_name} tool.
3. The expected_output must be a valid JSON object with "name" (the tool name) and "arguments" (a dict of argument key-value pairs).
4. All argument values in expected_output must be in English, even if the user spoke in {lang["name"]}.
5. Make examples diverse — vary the specific details (locations, names, amounts, topics) across examples.
6. Do NOT include any commentary, explanation, or markdown formatting. Output ONLY the JSON array below.

OUTPUT FORMAT — return ONLY a valid JSON array, no other text:
[
  {{
    "user_prompt": "<user utterance in {lang["name"]}>",
    "expected_output": {{"name": "{tool_name}", "arguments": {{<key-value pairs>}}}}
  }},
  ...
]

Generate exactly {n} examples now:"""


# ─── Generation Logic ─────────────────────────────────────────────────────────

def parse_examples(raw_text: str, tool_name: str, lang_code: str) -> list[dict]:
    """Extract and validate examples from model output."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", raw_text).strip()

    # Find the JSON array
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []

    try:
        examples = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    valid = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        if "user_prompt" not in ex or "expected_output" not in ex:
            continue
        out = ex["expected_output"]
        if not isinstance(out, dict):
            continue
        if out.get("name") != tool_name:
            continue
        if "arguments" not in out or not isinstance(out["arguments"], dict):
            continue
        ex["language"] = lang_code
        ex["tool_name"] = tool_name
        valid.append(ex)

    return valid


def generate_batch(lang_code: str, tool_name: str, n: int) -> list[dict]:
    """Call Command A to generate n examples for (lang, tool). Returns validated list."""
    prompt = build_generation_prompt(lang_code, tool_name, n)
    delay = RETRY_DELAY

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = co.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )
            raw = response.message.content[0].text
            examples = parse_examples(raw, tool_name, lang_code)
            if examples:
                return examples
            print(f"    ⚠ Attempt {attempt}: zero valid examples parsed. Retrying…")
        except Exception as e:
            print(f"    ⚠ Attempt {attempt} error: {e}. Retrying in {delay}s…")

        time.sleep(delay)
        delay *= 2

    print(f"    ✗ All {RETRY_LIMIT} attempts failed for ({lang_code}, {tool_name}). Skipping.")
    return []


def add_metadata(examples: list[dict], lang_code: str, tool_name: str) -> list[dict]:
    """Add ids and metadata to each example."""
    result = []
    for i, ex in enumerate(examples, 1):
        result.append({
            "id": f"{lang_code}_{tool_name}_{i:03d}",
            "language": lang_code,
            "tool_name": tool_name,
            "user_prompt": ex["user_prompt"],
            "expected_output": ex["expected_output"],
            "generated_by": MODEL,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        })
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(languages: list[str], test_mode: bool, resume: bool):
    all_examples = []
    total_generated = 0

    for lang_code in languages:
        lang_name = LANG_CONFIG[lang_code]["name"]
        lang_examples = []
        print(f"\n{'='*60}")
        print(f"  Language: {lang_name} ({lang_code.upper()})")
        print(f"{'='*60}")

        tools_for_lang = TOOL_NAMES if not test_mode else TOOL_NAMES[:1]
        n_per_batch = 1 if test_mode else EXAMPLES_PER_BATCH

        for tool_name in tools_for_lang:
            raw_path = RAW_DIR / f"{lang_code}_{tool_name}.jsonl"

            # Resume: skip if already done
            if resume and raw_path.exists():
                existing = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                if len(existing) >= n_per_batch:
                    print(f"  ✔ [{tool_name}] already done ({len(existing)} examples). Skipping.")
                    lang_examples.extend(existing)
                    total_generated += len(existing)
                    continue

            print(f"  ▶ Generating {n_per_batch}× [{tool_name}]…", end=" ", flush=True)
            examples = generate_batch(lang_code, tool_name, n_per_batch)
            examples = add_metadata(examples, lang_code, tool_name)

            # Pad if we got fewer than requested (re-run once more)
            if len(examples) < n_per_batch and not test_mode:
                deficit = n_per_batch - len(examples)
                print(f"(got {len(examples)}, topping up {deficit})…", end=" ", flush=True)
                extra = generate_batch(lang_code, tool_name, deficit)
                extra = add_metadata(extra, lang_code, tool_name)
                # Re-id the extras to avoid collisions
                for j, ex in enumerate(extra, len(examples) + 1):
                    ex["id"] = f"{lang_code}_{tool_name}_{j:03d}"
                examples.extend(extra)

            # Save raw batch
            with open(raw_path, "w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            print(f"✓ {len(examples)} saved")
            lang_examples.extend(examples)
            total_generated += len(examples)

        print(f"\n  ✔ {lang_name} total: {len(lang_examples)} examples")
        all_examples.extend(lang_examples)

    # Shuffle and write final dataset
    random.shuffle(all_examples)
    with open(FINAL_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"  ✅ DONE — {total_generated} examples → {FINAL_PATH}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Tiny Facade Dataset Generator")
    parser.add_argument("--test", action="store_true", help="Generate 1 example per language (3 total) for testing")
    parser.add_argument("--lang", choices=["en", "sw", "lg"], help="Generate one language only")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed batches")
    args = parser.parse_args()

    if args.lang:
        languages = [args.lang]
    else:
        languages = ["en", "sw", "lg"]

    mode = "TEST" if args.test else "FULL"
    print(f"\n🚀 Tiny Facade Dataset Generator")
    print(f"   Model: {MODEL}")
    print(f"   Mode: {mode}")
    print(f"   Languages: {', '.join(languages)}")
    if not args.test:
        target = EXAMPLES_PER_BATCH * TOOLS_COUNT * len(languages)
        print(f"   Target: ~{target} examples ({EXAMPLES_PER_BATCH}/tool × {TOOLS_COUNT} tools × {len(languages)} langs)")

    run(languages=languages, test_mode=args.test, resume=args.resume)


if __name__ == "__main__":
    main()
