"""
NLP-08: Complete NLP Analysis Consolidation  [NEW — v1]
========================================================
Task: "Complete NLP analysis of OCR output files —
       Repeat NLP 1 to 7 through the NLP Pipeline"
Deliverable: NER (consolidated, deduplicated, quality-scored)

Team: Abna (pipeline runner), Harini (NER QA), Neha (Relations QA)

What this module does:
  1. Reads ALL output from NLP-01 → NLP-07 (any number of files)
  2. Consolidates NER entities with frequency & coverage scores
  3. Consolidates relations with confidence ranking
  4. Produces a quality-scored NER report (ner_consolidated.xlsx)
  5. Produces a ranked relations report (relations_consolidated.csv)
  6. Produces a pipeline health summary (nlp08_health.json)
  7. Flags low-quality or suspicious entities for QA review

PRC Pramana Interns 2026
"""

import json
import re
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

# ── Label display order for NER export ──────────────────────────────
LABEL_ORDER = [
    "HERB", "FORMULATION", "DISEASE", "DOSHA", "PROCEDURE",
    "BODY_PART", "PROPERTY", "SOURCE_REF", "PLANT_PART",
    "QUANTITY", "INGREDIENT", "DIET",
]

# ── Confidence ranking helper ────────────────────────────────────────
CONF_RANK = {"high": 3, "medium": 2, "low": 1, "": 0}


# ──────────────────────────────────────────────────────────────────── #
#  Entity quality scorer                                               #
# ──────────────────────────────────────────────────────────────────── #

def entity_quality_score(entity_text: str, label: str, frequency: int) -> dict:
    """
    Returns a quality dict:
      score      : 0-100 (higher = more reliable for QA)
      flags      : list of issues to review
    """
    flags = []
    score = 60  # base

    # Frequency bonus
    if frequency >= 5:
        score += 20
    elif frequency >= 2:
        score += 10

    # Length check
    text_len = len(entity_text.strip())
    if text_len <= 2:
        flags.append("VERY_SHORT")
        score -= 20
    elif text_len >= 50:
        flags.append("VERY_LONG")
        score -= 10

    # Label-specific checks
    if label == "QUANTITY":
        if not re.search(r"\d", entity_text):
            flags.append("QUANTITY_NO_DIGIT")
            score -= 15

    if label in ("HERB", "DISEASE", "PROCEDURE"):
        # Should not be all digits
        if re.match(r"^\d+$", entity_text.strip()):
            flags.append("ENTITY_IS_PURE_NUMBER")
            score -= 30

    # Stop-word check
    stop_words = {"the", "a", "an", "is", "are", "of", "in", "at", "on", "for"}
    if entity_text.strip().lower() in stop_words:
        flags.append("STOP_WORD")
        score -= 40

    score = max(0, min(100, score))
    return {"score": score, "flags": flags, "needs_review": score < 50 or len(flags) > 0}


# ──────────────────────────────────────────────────────────────────── #
#  Main NLP-08 class                                                   #
# ──────────────────────────────────────────────────────────────────── #

class NLPConsolidator:
    """
    NLP-08: Reads all pipeline output and builds consolidated QA-ready exports.
    Can be run standalone (after run_pipeline.py) or from the Streamlit app.
    """

    def __init__(self, input_files, output_dir, logger, lang_hint="auto", prev_outputs=None):
        self.input_files  = input_files       # original OCR files (passed for compatibility)
        self.output_dir   = Path(output_dir)
        self.logger       = logger
        self.lang_hint    = lang_hint
        self.prev_outputs = prev_outputs or {}

        # Root pipeline output dir (parent of NLP-08 folder)
        self.pipeline_root = self.output_dir.parent

    # ── Entity collection ────────────────────────────────────────────

    def collect_all_entities(self) -> list:
        """
        Gather entities from NLP-03 (_annotated.json) and NLP-05 (_ner.json).
        Returns list of entity dicts with source tracking.
        """
        rows = []
        seen_key_sources: dict = defaultdict(set)  # (text_lower, label) → set of source files

        patterns_to_keys = [
            ("*_annotated.json", "annotations"),
            ("*_ner.json",       "entities"),
        ]

        for file_glob, data_key in patterns_to_keys:
            for json_file in self.pipeline_root.rglob(file_glob):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    source_file = data.get("source_file", json_file.name)
                    items = data.get(data_key, [])
                    for item in items:
                        entity_text = item.get("text", "").strip()
                        label       = item.get("label", "UNKNOWN")
                        if not entity_text:
                            continue
                        key = (entity_text.lower(), label)
                        seen_key_sources[key].add(source_file)
                        rows.append({
                            "Entity":      entity_text,
                            "Label":       label,
                            "Source_File": source_file,
                            "Start":       item.get("start", ""),
                            "End":         item.get("end", ""),
                        })
                except Exception as e:
                    self.logger.warning(f"  [NLP-08] Could not read {json_file.name}: {e}")

        # Add frequency and quality score
        freq_map = defaultdict(int)
        for (text_lower, label), sources in seen_key_sources.items():
            freq_map[(text_lower, label)] = len(sources)

        enriched = []
        seen_final = set()
        for row in rows:
            key = (row["Entity"].lower(), row["Label"])
            if key in seen_final:
                continue
            seen_final.add(key)
            freq = freq_map[key]
            quality = entity_quality_score(row["Entity"], row["Label"], freq)
            enriched.append({
                **row,
                "Frequency":    freq,
                "Quality_Score": quality["score"],
                "Flags":        "|".join(quality["flags"]) if quality["flags"] else "",
                "Needs_Review": quality["needs_review"],
            })

        return enriched

    # ── Relation collection ──────────────────────────────────────────

    def collect_all_relations(self) -> list:
        """
        Gather relations from NLP-07 (_relations.json).
        Returns list of relation dicts, deduplicated, highest confidence kept.
        """
        all_rels = []
        best: dict = {}

        for json_file in self.pipeline_root.rglob("*_relations.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                source_file = data.get("source_file", json_file.name)
                for rel in data.get("relations", []):
                    if rel.get("relation") == "CO_OCCURS_WITH":
                        continue
                    key = (
                        rel.get("relation", ""),
                        rel.get("subject", "").strip().lower(),
                        rel.get("object",  "").strip().lower(),
                    )
                    existing_conf = CONF_RANK.get(best.get(key, {}).get("Confidence", ""), 0)
                    new_conf      = CONF_RANK.get(rel.get("confidence", ""), 0)
                    if new_conf >= existing_conf:
                        best[key] = {
                            "Source":      rel.get("subject", "").strip(),
                            "Relation":    rel.get("relation", ""),
                            "Target":      rel.get("object",  "").strip(),
                            "Confidence":  rel.get("confidence", "low"),
                            "Context":     rel.get("context", "")[:120],
                            "Source_File": source_file,
                        }
            except Exception as e:
                self.logger.warning(f"  [NLP-08] Could not read {json_file.name}: {e}")

        return list(best.values())

    # ── Pipeline health check ────────────────────────────────────────

    def health_check(self, entities: list, relations: list) -> dict:
        """Produce a health summary: what ran, counts, potential issues."""
        label_counts = defaultdict(int)
        for e in entities:
            label_counts[e["Label"]] += 1

        rel_type_counts = defaultdict(int)
        conf_counts = defaultdict(int)
        for r in relations:
            rel_type_counts[r["Relation"]] += 1
            conf_counts[r["Confidence"]] += 1

        needs_review = [e for e in entities if e.get("Needs_Review")]

        issues = []
        if label_counts.get("HERB", 0) == 0:
            issues.append("WARNING: No HERB entities found — check NLP-03 seed dict or input files")
        if label_counts.get("DISEASE", 0) == 0:
            issues.append("WARNING: No DISEASE entities found")
        if len(relations) == 0:
            issues.append("WARNING: No relations extracted — check NLP-07 ran successfully")
        if conf_counts.get("high", 0) == 0 and len(relations) > 0:
            issues.append("INFO: No high-confidence relations — consider upgrading to dependency parser")
        if len(needs_review) > len(entities) * 0.3:
            issues.append(f"WARNING: {len(needs_review)} entities ({len(needs_review)*100//max(1,len(entities))}%) flagged for review — check annotation quality")

        # Stages run detection
        stages_found = []
        for stage_dir in self.pipeline_root.iterdir():
            if stage_dir.is_dir() and stage_dir.name.startswith("nlp_"):
                stages_found.append(stage_dir.name)

        return {
            "total_entities":        len(entities),
            "total_relations":       len(relations),
            "entity_label_counts":   dict(label_counts),
            "relation_type_counts":  dict(rel_type_counts),
            "relation_confidence":   dict(conf_counts),
            "entities_need_review":  len(needs_review),
            "pipeline_stages_found": sorted(stages_found),
            "issues":                issues,
            "status": "OK" if not any("WARNING" in i for i in issues) else "NEEDS_ATTENTION",
        }

    # ── Export helpers ───────────────────────────────────────────────

    def export_ner_xlsx(self, entities: list, out_path: Path):
        if not PANDAS_OK:
            self.logger.error("  pandas not installed — cannot export XLSX")
            return None

        if not entities:
            df = pd.DataFrame(columns=["Entity", "Label", "Source_File", "Frequency", "Quality_Score", "Flags", "Needs_Review"])
        else:
            df = pd.DataFrame(entities)
            # Column order: QA-critical cols first
            cols = ["Entity", "Label", "Source_File", "Frequency", "Quality_Score", "Needs_Review", "Flags", "Start", "End"]
            df = df[[c for c in cols if c in df.columns]]
            # Sort: by Label (canonical order) then by Quality_Score descending
            label_rank = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
            df["_label_rank"] = df["Label"].map(lambda x: label_rank.get(x, 99))
            df = df.sort_values(["_label_rank", "Quality_Score"], ascending=[True, False])
            df = df.drop(columns=["_label_rank"]).reset_index(drop=True)

        df.to_excel(out_path, index=False)
        self.logger.info(f"  ✅ Consolidated NER → {out_path}  ({len(df)} entities)")
        return df

    def export_relations_csv(self, relations: list, out_path: Path):
        if not PANDAS_OK:
            self.logger.error("  pandas not installed — cannot export CSV")
            return None

        if not relations:
            df = pd.DataFrame(columns=["Source", "Relation", "Target", "Confidence", "Context", "Source_File"])
        else:
            df = pd.DataFrame(relations)
            cols = ["Source", "Relation", "Target", "Confidence", "Context", "Source_File"]
            df = df[[c for c in cols if c in df.columns]]
            # Sort: by confidence tier (high first) then relation type
            conf_order = {"high": 0, "medium": 1, "low": 2}
            df["_conf_rank"] = df["Confidence"].map(lambda x: conf_order.get(x, 3))
            df = df.sort_values(["_conf_rank", "Relation"]).drop(columns=["_conf_rank"]).reset_index(drop=True)

        df.to_csv(out_path, index=False)
        self.logger.info(f"  ✅ Consolidated Relations → {out_path}  ({len(df)} relations)")
        return df

    # ── Runner ───────────────────────────────────────────────────────

    def run(self) -> dict:
        self.logger.info("  NLP-08: Starting consolidation of all pipeline output…")

        entities  = self.collect_all_entities()
        relations = self.collect_all_relations()
        health    = self.health_check(entities, relations)

        # Save health report
        health_path = self.output_dir / "nlp08_health.json"
        health_path.write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info(f"  Health status: {health['status']}")
        for issue in health.get("issues", []):
            self.logger.warning(f"    {issue}")

        # Export consolidated files
        ner_path = self.output_dir / "ner_consolidated.xlsx"
        rel_path = self.output_dir / "relations_consolidated.csv"
        ner_df = self.export_ner_xlsx(entities, ner_path)
        rel_df = self.export_relations_csv(relations, rel_path)

        # Also copy to qa_exports for backward compatibility
        qa_dir = self.pipeline_root / "qa_exports"
        qa_dir.mkdir(exist_ok=True)
        if ner_path.exists():
            import shutil
            shutil.copy(ner_path, qa_dir / "ner_output.xlsx")
            shutil.copy(rel_path, qa_dir / "relations_output.csv")
            self.logger.info(f"  Copied to qa_exports/")

        summary = (
            f"{len(entities)} entities consolidated, "
            f"{len(relations)} relations consolidated, "
            f"health: {health['status']}"
        )

        return {
            "task":             "NLP-08",
            "output_files":     [ner_path, rel_path, health_path],
            "total_entities":   len(entities),
            "total_relations":  len(relations),
            "health":           health,
            "summary":          summary,
        }


# ── Standalone runner ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="NLP-08: Consolidate pipeline output")
    parser.add_argument("--pipeline-output", "-p", default="./output",
                        help="Root output folder from run_pipeline.py")
    parser.add_argument("--out", "-o", default=None,
                        help="Where to save NLP-08 output (default: <pipeline-output>/nlp_08_consolidation)")
    args = parser.parse_args()

    pipeline_root = Path(args.pipeline_output)
    out_dir = Path(args.out) if args.out else pipeline_root / "nlp_08_consolidation"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("nlp08")

    consolidator = NLPConsolidator(
        input_files=[],
        output_dir=out_dir,
        logger=logger,
    )
    result = consolidator.run()

    print(f"\n✅ NLP-08 done: {result['summary']}")
    print(f"   NER Excel   → {out_dir}/ner_consolidated.xlsx  (for Harini)")
    print(f"   Relations   → {out_dir}/relations_consolidated.csv  (for Neha)")
    print(f"   Health      → {out_dir}/nlp08_health.json")