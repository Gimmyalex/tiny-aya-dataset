"""
MASSIVE to Tool-Call Converter (Task 1.2)
========================================
Converts MASSIVE dataset (intent/slot) into Tiny Facade tool-calling format (JSON).
Supports Swahili (sw-KE) and English (en-US).
"""

import json, re, os
from pathlib import Path

# ─── Mapping Config ──────────────────────────────────────────────────────────

INTENT_MAP = {
    "weather_query": "get_weather",
    "qa_factoid": "search_web",
    "qa_maths": "calculate",
    "alarm_set": "set_reminder",
    "news_query": "get_news",
    "qa_currency": "convert_currency",
    "email_sendemail": "send_message",
    "calendar_set": "create_event",
    "transport_query": "get_directions",
    "datetime_query": "search_web", # fallback
    "qa_definition": "search_web",
}

SLOT_MAP = {
    "place_name": "location",
    "person": "recipient",
    "email_body": "message",
    "news_topic": "topic",
    "time": "datetime",
    "date": "datetime", # will combine in code
    "event_name": "title",
    "currency_name": "to_currency", # rough mapping
    "object": "query",
}

# ─── Parsing Logic ───────────────────────────────────────────────────────────

def parse_annotated_utt(annot_utt):
    """
    Extracts slots from MASSIVE annot_utt.
    Format: [label : entity]
    Example: 'wake me up at [time : five am]' -> {'time': 'five am'}
    """
    slots = {}
    pattern = r"\[\s*(\w+)\s*:\s*([^\]]+)\]"
    matches = re.finditer(pattern, annot_utt)
    for match in matches:
        label, value = match.groups()
        slots[label.strip()] = value.strip()
    return slots

def build_tool_call(intent, slots, utt):
    tool_name = INTENT_MAP.get(intent)
    if not tool_name:
        return None

    args = {}
    
    # Simple mapping
    for massive_slot, value in slots.items():
        arg_name = SLOT_MAP.get(massive_slot)
        if arg_name:
            # Handle combining date/time for datetime
            if arg_name == "datetime" and "datetime" in args:
                args["datetime"] = f"{args['datetime']} {value}"
            else:
                args[arg_name] = value

    # Intent-specific adjustments
    if tool_name == "get_weather":
        if "location" not in args: args["location"] = "current location"
        args["unit"] = "celsius"
    
    elif tool_name == "calculate":
        # Extract numbers or the whole utt if it's mathy
        args["expression"] = slots.get("number", utt)
        
    elif tool_name == "search_web":
        if "query" not in args: args["query"] = utt
        
    elif tool_name == "convert_currency":
        args["amount"] = 1.0 # default if not found
        if "from_currency" not in args: args["from_currency"] = "USD"
        if "to_currency" not in args: args["to_currency"] = "UGX"

    elif tool_name == "get_news":
        if "topic" not in args: args["topic"] = "top headlines"

    # Final check: make sure we have required args for the tool schema
    # (Simplified check)
    if tool_name == "send_message" and "recipient" not in args:
        args["recipient"] = slots.get("person", "Unknown")
        if "message" not in args: args["message"] = utt

    return {"name": tool_name, "arguments": args}

# ─── Main Processing ─────────────────────────────────────────────────────────

def process_file(lang_code, input_path, output_path):
    print(f"Processing {lang_code}...")
    count = 0
    with open(input_path, encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            intent = data["intent"]
            
            if intent in INTENT_MAP:
                slots = parse_annotated_utt(data["annot_utt"])
                tool_call = build_tool_call(intent, slots, data["utt"])
                
                if tool_call:
                    out_data = {
                        "id": f"massive_{lang_code}_{data['id']}",
                        "language": lang_code[:2],
                        "tool_name": tool_call["name"],
                        "user_prompt": data["utt"],
                        "expected_output": tool_call,
                        "metadata": {
                            "original_intent": intent,
                            "original_scenario": data["scenario"]
                        }
                    }
                    f_out.write(json.dumps(out_data, ensure_ascii=False) + "\n")
                    count += 1
    print(f"  ✔ Saved {count} examples to {output_path}")

def main():
    raw_dir = Path("massive_data")
    out_dir = Path("dataset/massive_converted")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    locales = ["en-US", "sw-KE"]
    for locale in locales:
        in_path = raw_dir / f"{locale}.jsonl"
        out_path = out_dir / f"{locale}_toolcalls.jsonl"
        if in_path.exists():
            process_file(locale, in_path, out_path)

if __name__ == "__main__":
    main()
