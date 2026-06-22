"""
export_for_qa.py
================
Converts NLP pipeline output → exact format the QA Streamlit tool expects.

QA-02 (NER Validator) expects:   Entity, Label          → ner_output.xlsx
QA-03 (Relationship)  expects:   Source, Relation, Target → relations_output.csv

Run this after run_pipeline.py:
    python export_for_qa.py --output ./output
"""

import json
import argparse
from pathlib import Path

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False
    print("[ERROR] pandas not installed. Run: pip install pandas openpyxl")
    exit(1)


def collect_ner_rows(output_dir: Path) -> list:
    """
    Pull entity rows from NLP-03 annotated JSONs and NLP-05 NER JSONs.
    Returns list of {Entity, Label, Source_File} dicts.
    """
    rows = []
    seen = set()

    # Search NLP-03 annotation output
    for json_file in output_dir.rglob("*_annotated.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            for ann in data.get("annotations", []):
                key = (ann["text"].strip().lower(), ann["label"])
                if key not in seen:
                    seen.add(key)
                    rows.append({
                        "Entity":      ann["text"].strip(),
                        "Label":       ann["label"],
                        "Source_File": data.get("source_file", json_file.name),
                        "Start":       ann.get("start", ""),
                        "End":         ann.get("end", ""),
                    })
        except Exception as e:
            print(f"  [WARN] Could not read {json_file.name}: {e}")

    # Also pull from NLP-05 NER output (may have additional entities)
    for json_file in output_dir.rglob("*_ner.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            for ent in data.get("entities", []):
                key = (ent["text"].strip().lower(), ent["label"])
                if key not in seen:
                    seen.add(key)
                    rows.append({
                        "Entity":      ent["text"].strip(),
                        "Label":       ent["label"],
                        "Source_File": data.get("source_file", json_file.name),
                        "Start":       ent.get("start", ""),
                        "End":         ent.get("end", ""),
                    })
        except Exception as e:
            print(f"  [WARN] Could not read {json_file.name}: {e}")

    return rows


def collect_relation_rows(output_dir: Path) -> list:
    """
    Pull relation rows from NLP-07 relation JSONs.
    Returns list of {Source, Relation, Target, Confidence, Source_File} dicts.
    """
    rows = []
    seen = set()

    for json_file in output_dir.rglob("*_relations.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            for rel in data.get("relations", []):
                # Skip weak co-occurrence unless nothing else exists
                if rel.get("relation") == "CO_OCCURS_WITH":
                    continue
                key = (
                    rel["subject"].strip().lower(),
                    rel["relation"],
                    rel["object"].strip().lower(),
                )
                if key not in seen:
                    seen.add(key)
                    rows.append({
                        "Source":      rel["subject"].strip(),
                        "Relation":    rel["relation"],
                        "Target":      rel["object"].strip(),
                        "Confidence":  rel.get("confidence", "medium"),
                        "Context":     rel.get("context", "")[:100],
                        "Source_File": data.get("source_file", json_file.name),
                    })
        except Exception as e:
            print(f"  [WARN] Could not read {json_file.name}: {e}")

    return rows


def export_ner_xlsx(rows: list, out_path: Path):
    """Save NER rows as Excel — QA-02 format: Entity, Label columns first."""
    if not rows:
        print("  [WARN] No NER entities found to export.")
        df = pd.DataFrame(columns=["Entity", "Label", "Source_File", "Start", "End"])
    else:
        df = pd.DataFrame(rows)
        # QA tool only strictly needs Entity + Label — keep them first
        cols = ["Entity", "Label", "Source_File", "Start", "End"]
        df = df[[c for c in cols if c in df.columns]]
        df = df.sort_values(["Label", "Entity"]).reset_index(drop=True)

    df.to_excel(out_path, index=False)
    print(f"  ✅ NER export → {out_path}  ({len(df)} entities)")
    return df


def export_relations_csv(rows: list, out_path: Path):
    """Save relation rows as CSV — QA-03 format: Source, Relation, Target columns first."""
    if not rows:
        print("  [WARN] No relations found to export.")
        df = pd.DataFrame(columns=["Source", "Relation", "Target", "Confidence", "Context", "Source_File"])
    else:
        df = pd.DataFrame(rows)
        cols = ["Source", "Relation", "Target", "Confidence", "Context", "Source_File"]
        df = df[[c for c in cols if c in df.columns]]
        df = df.sort_values(["Relation", "Source"]).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"  ✅ Relations export → {out_path}  ({len(df)} relations)")
    return df


def print_summary(ner_df: pd.DataFrame, rel_df: pd.DataFrame):
    print("\n── NER Summary ─────────────────────────────────")
    if not ner_df.empty and "Label" in ner_df.columns:
        for label, count in ner_df["Label"].value_counts().items():
            print(f"   {label:<20} {count}")
    else:
        print("   (no entities)")

    print("\n── Relations Summary ───────────────────────────")
    if not rel_df.empty and "Relation" in rel_df.columns:
        for rel, count in rel_df["Relation"].value_counts().items():
            print(f"   {rel:<25} {count}")
    else:
        print("   (no relations)")
    print("────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export NLP pipeline output → QA team format"
    )
    parser.add_argument("--output", "-o", default="./output",
                        help="Root output folder from run_pipeline.py (default: ./output)")
    parser.add_argument("--qa-folder", "-q", default="./qa_exports",
                        help="Where to save QA export files (default: ./qa_exports)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    qa_dir     = Path(args.qa_folder)
    qa_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Reading pipeline output from: {output_dir}")
    print(f"📤 Saving QA exports to:         {qa_dir}\n")

    # Collect
    ner_rows = collect_ner_rows(output_dir)
    rel_rows = collect_relation_rows(output_dir)

    # Export
    ner_df = export_ner_xlsx(ner_rows, qa_dir / "ner_output.xlsx")
    rel_df = export_relations_csv(rel_rows, qa_dir / "relations_output.csv")

    print_summary(ner_df, rel_df)

    print("✅ Done. Hand these two files to the QA team:")
    print(f"   → {qa_dir / 'ner_output.xlsx'}       (for Harini — Tab 2)")
    print(f"   → {qa_dir / 'relations_output.csv'}  (for Neha   — Tab 3 & 4)\n")


if __name__ == "__main__":
    main()
