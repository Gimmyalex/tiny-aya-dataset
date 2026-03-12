# Tiny Facade: Multilingual Tool-Calling Dataset Generation

This project is part of the **Expedition Tiny Aya** initiative, focused on creating high-quality synthetic and localized datasets for fine-tuning compact LLMs on multilingual tool-calling (function calling) tasks.

## 🚀 Overview

The goal of this module (Task 1.2) was to generate a 600-example dataset covering **English (EN)**, **Swahili (SW)**, and **Luganda (LG)**. 

Following a strategic pivot based on the **MASSIVE-Agents** research (EMNLP 2025), we moved away from purely synthetic generation to a human-grounded pipeline using the MASSIVE dataset (Amazon AGI) as a foundation.

## 📊 Dataset Statistics

The final training dataset contains **597 examples** balanced across intent categories:

| Language | Source | Count |
|---|---|---|
| **English** | MASSIVE (en-US) | 200 |
| **Swahili** | MASSIVE (sw-KE) | 200 |
| **Luganda** | Gemini-Translated (EN → LG) | 197 |

**Core Tools Covered:**
- `get_weather`, `search_web`, `calculate`, `set_reminder`, `get_news`, `convert_currency`, `send_message`, `create_event`, `get_directions`.

## 🛠 Project Structure

- `dataset/tool_calling_dataset.jsonl`: The final multilingual training file.
- `massive_to_toolcall.py`: Maps MASSIVE intents/slots to Tiny Facade's tool JSON schema.
- `translate_to_luganda.py`: Automated translation of EN utterances into natural Luganda using the `google-genai` SDK.
- `merge_and_validate.py`: Merges language splits, validates JSON integrity, and prints dataset statistics.
- `compare_models.py`: Evaluation script used to choose the best LLM for data generation (Command A wins).

## ⚙️ Setup & Usage

### 1. Requirements
- Python 3.10+
- `cohere` SDK (for synthetic generation/comparison)
- `google-genai` SDK (for Luganda translation)
- `datasets` (for MASSIVE access)
- `python-dotenv`

### 2. Installation
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install cohere google-genai datasets python-dotenv
```

### 3. Environment
Create a `.env` file in the root:
```env
COHERE_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

### 4. Running the Pipeline
To reproduce the dataset from scratch:
1. **Prepare Data:** Run `massive_to_toolcall.py` (downloads and converts MASSIVE).
2. **Translate:** Run `translate_to_luganda.py` (requires Gemini key).
3. **Assemble:** Run `merge_and_validate.py` to create the final `tool_calling_dataset.jsonl`.

## 🔬 Research Findings: Model Comparison

We benchmarked two Cohere models for synthetic data quality:
- **Command A (Standard):** 🏆 Selected. Correctly maintains argument values in English while the prompt is in Luganda.
- **Command A Translate:** Rejected. Attempted to translate JSON argument values (e.g., city names) into Luganda, which breaks tool execution logic.

---
*Created for the Tiny Aya Expedition by the Tiny Facade Team.*
