# Ayurveda NLP Pipeline — Streamlit App
### PRC Pramana Interns 2026

A full-featured web UI that runs the NLP-01→NLP-07 pipeline on uploaded OCR text files.

---

## Features

| Tab | What it does |
|-----|-------------|
| **▶ Run Pipeline** | Upload `.txt` files, select tasks, run the full pipeline, view logs & stats |
| **🔬 NER Entities** | Browse, filter, and search all extracted entities with label badges |
| **🔗 Relations** | View TREATS / BALANCES_DOSHA / CONTAINS relations as visual cards |
| **📤 QA Export** | Download `ner_output.xlsx` (Harini) and `relations_output.csv` (Neha) |

---

## Local Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download spaCy model
python -m spacy download xx_ent_wiki_sm

# 3. Launch the app
streamlit run app.py
```

Opens at → http://localhost:8501

---

## Deploy to Streamlit Cloud (free)

1. Push this folder to a **GitHub repo** (public or private)

2. Go to **https://share.streamlit.io** → "New app"

3. Set:
   - **Repository**: `your-org/your-repo`
   - **Branch**: `main`
   - **Main file path**: `app.py`

4. Click **Deploy** — done! Streamlit Cloud installs `requirements.txt` automatically.

> **Note**: The optional `transformers` / `torch` / `stanza` lines in `requirements.txt` are commented out to keep the cloud build fast. Uncomment them for full Indic NLP support (adds ~2 GB to image).

---

## File Structure

```
ayurveda_nlp_app/
├── app.py                    ← Streamlit UI (this is your entry point)
├── run_pipeline.py           ← Pipeline orchestrator
├── export_for_qa.py          ← QA export helper
├── nlp01_text_cleaning.py
├── nlp02_language_detect.py
├── nlp03_annotation.py
├── nlp05_ner.py
├── nlp06_semantic_tagging.py
├── nlp07_relationship.py
├── utils/
│   ├── __init__.py
│   ├── logger.py             ← Logger (with Streamlit-compatible handler)
│   └── report.py             ← PipelineReport
├── requirements.txt
├── packages.txt              ← System deps for Streamlit Cloud
└── .streamlit/
    └── config.toml           ← Theme + server config
```

---

## QA Team Handoff

After each run, the **📤 QA Export** tab provides:
- `ner_output.xlsx` → **Harini** (Tab 2 · NER Validator in QA Streamlit tool)  
- `relations_output.csv` → **Neha** (Tab 3 & 4 · Relationship in QA Streamlit tool)
