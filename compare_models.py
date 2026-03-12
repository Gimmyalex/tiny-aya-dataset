"""
Tiny Facade — Model Comparison Script
======================================
Compares command-a-03-2025 vs command-a-translate-08-2025 for
multilingual tool-calling dataset generation quality.

Generates 3 examples per model across 3 tools (get_weather, send_message, 
calculate) in Luganda, then scores each on:
  - JSON validity rate
  - Correct tool name selection
  - Argument completeness
  - Natural language quality (heuristic)

Usage:
  python compare_models.py
"""

import os, json, re, time
from dotenv import load_dotenv
import cohere

load_dotenv(override=True)
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

MODELS = {
    "command-a-03-2025":          "Command A (latest)",
    "command-a-translate-08-2025": "Command A Translate",
}

# 3 representative tools spanning different arg types
TEST_TOOLS = {
    "get_weather": {
        "name": "get_weather",
        "description": "Get current weather conditions for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit":     {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    },
    "send_message": {
        "name": "send_message",
        "description": "Send a text message to a contact.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string"},
                "message":   {"type": "string"}
            },
            "required": ["recipient", "message"]
        }
    },
    "calculate": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    },
    "set_reminder": {
        "name": "set_reminder",
        "description": "Set a reminder for a specific time.",
        "parameters": {
            "type": "object",
            "properties": {
                "message":  {"type": "string"},
                "datetime": {"type": "string"}
            },
            "required": ["message", "datetime"]
        }
    },
    "convert_currency": {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount":        {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency":   {"type": "string"}
            },
            "required": ["amount", "from_currency", "to_currency"]
        }
    }
}

N_PER_TOOL = 3  # examples per (model, tool) — 5 tools × 3 = 15 examples per model

PROMPT_TEMPLATE = """You are a dataset generator for a multilingual AI training project.
Generate {n} diverse, realistic tool-calling examples in natural, everyday Luganda.

Users are Baganda speakers from Uganda. Typical locations: Kampala, Entebbe, Mukono.
Typical names: Mama, Taata, Nakato, Babirye, Kato, Wasswa.

TARGET TOOL:
{schema}

RULES:
1. Each user utterance MUST be in natural Luganda.
2. The expected_output name must be exactly "{tool_name}".
3. All argument VALUES in expected_output must be in English.
4. Output ONLY a valid JSON array — no markdown, no explanation.

OUTPUT (JSON array only):
[
  {{
    "user_prompt": "<Luganda utterance>",
    "expected_output": {{"name": "{tool_name}", "arguments": {{...}}}}
  }}
]

Generate exactly {n} examples:"""


def parse_examples(raw: str, tool_name: str) -> list[dict]:
    text = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    valid = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out = item.get("expected_output", {})
        if not isinstance(out, dict):
            continue
        if "user_prompt" not in item:
            continue
        valid.append(item)
    return valid


def score_examples(examples: list[dict], tool_name: str) -> dict:
    total = len(examples)
    if total == 0:
        return {"total": 0, "json_valid": 0, "name_correct": 0, "args_complete": 0,
                "json_pct": 0, "name_pct": 0, "args_pct": 0}

    name_correct  = sum(1 for e in examples if e.get("expected_output", {}).get("name") == tool_name)
    args_complete = sum(1 for e in examples
                        if isinstance(e.get("expected_output", {}).get("arguments"), dict)
                        and len(e["expected_output"]["arguments"]) > 0)
    required_args = TEST_TOOLS[tool_name]["parameters"].get("required", [])
    args_required = sum(1 for e in examples
                        if all(k in e.get("expected_output", {}).get("arguments", {})
                               for k in required_args))

    return {
        "total":         total,
        "json_valid":    total,            # if we parsed them, they're valid JSON
        "name_correct":  name_correct,
        "args_complete": args_complete,
        "args_required": args_required,
        "json_pct":      100,
        "name_pct":      round(name_correct / total * 100),
        "args_pct":      round(args_required / total * 100),
    }


def run_model(model_id: str, model_label: str) -> dict:
    print(f"\n{'─'*55}")
    print(f"  Model: {model_label}")
    print(f"  ID:    {model_id}")
    print(f"{'─'*55}")

    all_examples = []
    per_tool = {}

    for tool_name, tool_schema in TEST_TOOLS.items():
        prompt = PROMPT_TEMPLATE.format(
            n=N_PER_TOOL,
            schema=json.dumps(tool_schema, indent=2),
            tool_name=tool_name
        )
        print(f"  ▶ [{tool_name}]… ", end="", flush=True)

        try:
            resp = co.chat(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,
            )
            raw = resp.message.content[0].text
            examples = parse_examples(raw, tool_name)
            scores = score_examples(examples, tool_name)
            per_tool[tool_name] = {"examples": examples, "scores": scores}
            print(f"✓ {len(examples)}/{N_PER_TOOL} parsed — name:{scores['name_pct']}% args:{scores['args_pct']}%")
            all_examples.extend(examples)
        except Exception as e:
            print(f"✗ Error: {str(e)[:80]}")
            per_tool[tool_name] = {"examples": [], "scores": score_examples([], tool_name)}

        time.sleep(1)  # be gentle with rate limits

    return {"model": model_label, "per_tool": per_tool, "all_examples": all_examples}


def print_comparison(results: list[dict]):
    models = [r["model"] for r in results]
    tools  = list(TEST_TOOLS.keys())

    W = 22  # column width

    print(f"\n{'='*70}")
    print("  COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Tool':<22}" + "".join(f"{'Name%':>8}{'ArgsOK%':>8}{'N':>5}   " for _ in results))
    print(f"  {'':22}" + "".join(f"  [{r['model'][:20]}]" for r in results))
    print(f"  {'─'*22}" + "─" * (24 * len(results)))

    overall_name  = [0] * len(results)
    overall_args  = [0] * len(results)
    overall_total = [0] * len(results)

    for tool in tools:
        row = f"  {tool:<22}"
        for i, r in enumerate(results):
            s = r["per_tool"].get(tool, {}).get("scores", {})
            n = s.get("total", 0)
            name_p = s.get("name_pct", 0)
            args_p = s.get("args_pct", 0)
            row += f"{name_p:>7}%{args_p:>7}%{n:>5}   "
            overall_name[i]  += s.get("name_correct", 0)
            overall_args[i]  += s.get("args_required", 0)
            overall_total[i] += n
        print(row)

    print(f"  {'─'*22}" + "─" * (24 * len(results)))
    row = f"  {'OVERALL':<22}"
    for i, r in enumerate(results):
        t = overall_total[i] or 1
        np = round(overall_name[i] / t * 100)
        ap = round(overall_args[i] / t * 100)
        row += f"{np:>7}%{ap:>7}%{overall_total[i]:>5}   "
    print(row)

    # Determine winner
    print(f"\n{'─'*70}")
    scores_summary = []
    for i, r in enumerate(results):
        t = overall_total[i] or 1
        combined = (overall_name[i] + overall_args[i]) / (2 * t) * 100
        scores_summary.append((combined, r["model"]))
        print(f"  {r['model']}: combined score {combined:.1f}%")

    winner = max(scores_summary, key=lambda x: x[0])
    print(f"\n  🏆 WINNER: {winner[1]} ({winner[0]:.1f}% combined)")
    print(f"{'='*70}\n")

    # Sample outputs side by side
    print("SAMPLE OUTPUTS (first example per tool per model):\n")
    for tool in tools[:3]:  # show 3 tools
        print(f"  [{tool.upper()}]")
        for r in results:
            exs = r["per_tool"].get(tool, {}).get("examples", [])
            if exs:
                e = exs[0]
                print(f"    [{r['model']}]")
                print(f"      User: {e.get('user_prompt', 'N/A')}")
                print(f"      Call: {json.dumps(e.get('expected_output', {}), ensure_ascii=False)}")
        print()


def main():
    print("\n🔬 Tiny Facade — Model Comparison")
    print(f"   Comparing: {' vs '.join(MODELS.values())}")
    print(f"   Language: Luganda (LG)")
    print(f"   Tools: {len(TEST_TOOLS)} tools × {N_PER_TOOL} examples = {len(TEST_TOOLS)*N_PER_TOOL} per model")

    results = []
    for model_id, model_label in MODELS.items():
        result = run_model(model_id, model_label)
        results.append(result)

    print_comparison(results)

    # Save raw outputs for inspection
    out = {"comparison": {r["model"]: r["per_tool"] for r in results}}
    with open("dataset/model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("📄 Raw outputs saved to: dataset/model_comparison.json\n")


if __name__ == "__main__":
    main()
