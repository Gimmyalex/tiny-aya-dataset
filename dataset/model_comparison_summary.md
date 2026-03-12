# Model Comparison: Command A (latest) vs Command A Translate
**Task 1.2 — Luganda Dataset Generation | Tiny Facade**
**Date:** March 12, 2026 | **Language:** Luganda (LG)

## Setup
- **5 tools tested:** get_weather, send_message, calculate, set_reminder, convert_currency
- **3 examples each** = 15 examples per model, 30 total
- Same prompts sent to both models

## Structural Scores (JSON validity / correct tool name / required args)

| Tool | Command A (latest) | Command A Translate |
|---|---|---|
| get_weather | 100% / 100% / 100% | 100% / 100% / 100% |
| send_message | 100% / 100% / 100% | 100% / 100% / 100% |
| calculate | 100% / 100% / 100% | 100% / 100% / 100% |
| set_reminder | 100% / 100% / 100% | 100% / 100% / 100% |
| convert_currency | 100% / 100% / 100% | 100% / 100% / 100% |
| **OVERALL** | **100% all metrics** | **100% all metrics** |

Both models scored perfectly on parse-able JSON, correct tool selection, and required argument coverage.

## Qualitative Comparison — Luganda Naturalness

### get_weather
| Model | Generated utterance |
|---|---|
| **Command A (latest)** | *"Mama, ekyuma mu Kampala kye ki?"* ✅ natural, everyday speech |
| Command A Translate | *"Mama, lina lya Kampala luli lwanyisa?"* ⚠️ slightly awkward phrasing |

### send_message — KEY DIFFERENTIATOR
| Model | message argument |
|---|---|
| **Command A (latest)** | `"message": "Do you want to come to the party?"` ✅ English (correct for training) |
| Command A Translate | `"message": "Ekyokya kye kya bulungi, nnooyi?"` ❌ Left message body in Luganda |

Command A Translate **broke the training rule** (all argument values must be in English for cross-lingual tool calling). This is a critical failure for our use case.

### calculate
| Model | Expression diversity |
|---|---|
| Command A (latest) | Simple: `20/7`, `15/3`, `12/4` — repetitive |
| **Command A Translate** | Complex: `(10000/4)+2500`, `(200+300)+50` ✅ better diversity |

### set_reminder
Command A Translate had slightly more varied Luganda sentence structures, but Command A (latest) produced correct English argument values consistently.

## Sample Side-by-Side

**Command A (latest):**
```json
{
  "user_prompt": "Babirye, weebereze amawunti 200,000 ga Uganda sshillingi mu pauni za Ugangale.",
  "expected_output": {"name": "convert_currency", "arguments": {"amount": 200000, "from_currency": "UGX", "to_currency": "GBP"}}
}
```

**Command A Translate:**
```json
{
  "user_prompt": "Nakato, 200,000 Usheringi biba biki mu Pound za Uingereza?",
  "expected_output": {"name": "convert_currency", "arguments": {"amount": 200000, "from_currency": "UGX", "to_currency": "GBP"}}
}
```
*(For convert_currency both are correct — the difference shows in send_message)*

## 🏆 Decision: Use `command-a-03-2025`

**Reason:** Command A Translate breaks the critical training rule of keeping argument values in English. For cross-lingual tool calling (user speaks Luganda → model outputs English JSON), this is a non-starter.

**Note:** The Luganda naturalness from both models should be validated by a native speaker (Kato). If quality is poor overall, fallback plan is MASSIVE EN + Gemini translation → LG (as discussed with Bronson).
