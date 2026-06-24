"""
NLP-07: Relationship Mining  [FIXED v4]
=========================================
What changed vs v3 (and why):

  PROBLEM (v3): `_read_text()` only handled two cases — the input file is
  already a .txt, or a single `rglob(stem + ".txt")` lookup. By the time
  NLP-07 runs, `current_inputs` are the JSON outputs of NLP-06, so it
  always falls into the rglob branch. If any earlier stage renamed the
  stem (e.g. a cleaning stage appending "_clean"), the rglob finds
  nothing and `text` silently becomes "". Proximity-based relations then
  extract zero results with no warning the user can act on, and
  co-occurrence relations (which also need `text` for sentence
  splitting) degrade to nothing as well.

  FIX (v4):
    1. _read_text() now searches in priority order:
         a. the file itself, if already .txt
         b. output_dir.parent/"input"/{stem}.txt   (the original
            extracted text — always present, never renamed, written
            directly by app.py's convert_uploaded_files_to_txt)
         c. NLP-01's output directory, matched by glob on stage name
            rather than a hard-coded path
         d. a broad rglob(f"{stem}*.txt") fallback that collects every
            candidate and keeps the longest (most complete) one instead
            of just "whichever rglob finds first"
       If nothing is found anywhere, it logs a specific warning naming
       the stem and the directory tree searched, instead of failing
       silently.

    2. cooccurrence_relations() now has a real fallback path
       (_cooccurrence_by_position) for when no source text exists at
       all: it sorts entities by character offset and groups ones close
       together by position instead of needing sentence boundaries from
       `text`. So even in the worst case (text genuinely unrecoverable)
       you still get low-confidence CO_OCCURS_WITH relations instead of
       zero relations.

  Everything else (entity loading via rglob across nlp_03/05/06 output,
  confidence scoring, trigger taxonomy, dedup) is unchanged from v3 —
  those parts were already correct.

Deliverable: NLP, Dependency Parsing
"""

import re
import json
from pathlib import Path
from itertools import combinations
from collections import defaultdict


# ── Relation patterns ────────────────────────────────────────────────

RELATION_PATTERNS = [
    ("TREATS",
     [
         (r"\b(treats?|cures?|heals?|eradicates?|eliminates?|destroys?)\b", "high"),
         (r"\b(alleviates?|relieves?|reduces?|diminishes?|controls?)\b", "high"),
         (r"\b(beneficial\s+for|useful\s+in|effective\s+(for|against)|indicated\s+for)\b", "medium"),
         (r"\b(used\s+(for|in|against|to\s+treat)|remedy\s+for)\b", "medium"),
         (r"\b(helps?\s+in|good\s+for|recommended\s+for)\b", "low"),
     ],
     ["HERB", "INGREDIENT", "PROCEDURE", "FORMULATION"],
     ["DISEASE", "BODY_PART"]),

    ("BALANCES_DOSHA",
     [
         (r"\b(balances?|pacifies?|normalizes?|harmonizes?)\b", "high"),
         (r"\b(reduces?|decreases?|alleviates?|mitigates?)\b", "high"),
         (r"\b(aggravates?|increases?|provokes?|vitiates?)\b", "high"),
         (r"\b(beneficial\s+for|controls?)\b", "medium"),
     ],
     ["HERB", "INGREDIENT", "PROCEDURE", "FORMULATION"],
     ["DOSHA"]),

    ("HAS_PLANT_PART",
     [
         (r"\b(from\s+the|using\s+the|prepared\s+from|extracted\s+from|derived\s+from)\b", "high"),
         (r"\b(part\s+used|used\s+part)\b", "high"),
         (r"\b(root|bark|leaf|leaves|fruit|seed|flower|resin)\s+(of|is|are)\b", "medium"),
     ],
     ["HERB"],
     ["PLANT_PART"]),

    ("AFFECTS_BODY",
     [
         (r"\b(affects?|involves?|attacks?|damages?|inflames?)\b", "high"),
         (r"\b(manifests?\s+in|located\s+in|found\s+in|present\s+in)\b", "medium"),
         (r"\b(of\s+the|in\s+the|at\s+the)\b", "low"),
     ],
     ["DISEASE"],
     ["BODY_PART"]),

    ("HAS_PROPERTY",
     [
         (r"\b(has|have|possesses?|exhibits?|shows?|displays?)\b", "high"),
         (r"\b(is\s+known\s+for|characterized\s+by|attributed\s+with)\b", "high"),
         (r"\b(is|are)\b", "low"),
     ],
     ["HERB", "INGREDIENT", "FORMULATION"],
     ["PROPERTY"]),

    ("MENTIONED_IN",
     [
         (r"\b(mentioned\s+in|described\s+in|cited\s+in|found\s+in)\b", "high"),
         (r"\b(according\s+to|as\s+per|as\s+stated\s+in)\b", "high"),
         (r"\b(in\s+the|from\s+the)\b", "low"),
     ],
     ["HERB", "DISEASE", "PROCEDURE", "FORMULATION"],
     ["SOURCE_REF"]),

    ("DOSAGE",
     [
         (r"\b(dose|dosage|taken|administered|given|used\s+at|prescribed\s+at)\b", "high"),
         (r"\b(in\s+a\s+dose|at\s+a\s+dose)\b", "high"),
         (r"\b(daily|twice|thrice|morning|evening)\b", "medium"),
     ],
     ["HERB", "INGREDIENT", "FORMULATION"],
     ["QUANTITY"]),

    ("INDICATED_FOR",
     [
         (r"\b(indicated\s+for|recommended\s+for|prescribed\s+for)\b", "high"),
         (r"\b(used\s+in|helpful\s+in|beneficial\s+in)\b", "medium"),
     ],
     ["PROCEDURE"],
     ["DISEASE"]),

    ("CONTAINS",
     [
         (r"\b(contains?|includes?|composed\s+of|made\s+(of|from|with)|consists?\s+of)\b", "high"),
         (r"\b(mixture\s+of|combination\s+of|compound\s+of)\b", "high"),
         (r"\b(with|and|plus)\b", "low"),
     ],
     ["HERB", "INGREDIENT", "FORMULATION"],
     ["HERB", "INGREDIENT"]),
]

PROXIMITY_WINDOW = 350
COOCCUR_POSITION_WINDOW = 200  # used only by the text-less fallback
CONF_RANK = {"high": 3, "medium": 2, "low": 1, "": 0}


def score_confidence(trigger_strength: str, proximity: int) -> str:
    if trigger_strength == "high"   and proximity < 150: return "high"
    if trigger_strength == "high"   and proximity < 300: return "medium"
    if trigger_strength == "medium" and proximity < 150: return "medium"
    return "low"


class RelationshipExtractor:
    def __init__(self, input_files, output_dir, logger,
                 lang_hint="auto", prev_outputs=None):
        self.input_files  = input_files
        self.output_dir   = Path(output_dir)
        self.logger       = logger
        self.lang_hint    = lang_hint
        self.prev_outputs = prev_outputs or {}

        self.compiled_patterns = [
            (rel_name,
             [(re.compile(pat, re.IGNORECASE | re.UNICODE), strength)
              for pat, strength in trigger_list],
             subj_labels, obj_labels)
            for rel_name, trigger_list, subj_labels, obj_labels in RELATION_PATTERNS
        ]

    # ── Entity loading ────────────────────────────────────────────────

    def load_entities(self, source_stem: str) -> list:
        """
        Uses rglob on the pipeline root — finds entity files regardless
        of what directory they live in. Priority: semantic > NER > annotated
        (most enriched first). Supports entities under any of the three
        key names different upstream stages use.
        """
        pipeline_root = self.output_dir.parent

        for suffix in ["_semantic.json", "_ner.json", "_annotated.json"]:
            for p in pipeline_root.rglob(source_stem + suffix):
                try:
                    data     = json.loads(p.read_text(encoding="utf-8"))
                    entities = data.get(
                        "entities",
                        data.get("tagged_entities",
                        data.get("annotations", []))
                    )
                    if entities:
                        self.logger.debug(
                            f"  NLP-07: loaded {len(entities)} entities "
                            f"from {p.relative_to(pipeline_root)}"
                        )
                        return entities
                except Exception as e:
                    self.logger.debug(f"  Could not read {p}: {e}")

        self.logger.warning(
            f"  NLP-07: no entities found for '{source_stem}'. "
            "Make sure NLP-03 / NLP-05 / NLP-06 ran successfully."
        )
        return []

    def _get_text_stem(self, f: Path) -> str:
        """Strip NLP-stage suffixes to get original file stem."""
        stem = f.stem
        for suf in ("_semantic", "_ner", "_annotated"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        return stem

    def _read_text(self, f: Path, stem: str) -> str:
        """
        Locate the best available source text for `stem`.

        Search order (stops at first non-empty result):
          1. `f` itself, if it's already a .txt file
          2. output_dir.parent/"input"/{stem}.txt — the original
             extracted text written by app.py before any NLP stage runs.
             This is the most reliable source: it's never renamed and
             always exists if the pipeline started correctly.
          3. NLP-01's output directory (matched by glob on the stage
             name pattern "nlp_01*", not a hard-coded path) — covers the
             case where cleaning meaningfully changed the text and we'd
             rather extract relations from the cleaned version.
          4. A broad rglob(f"{stem}*.txt") across the whole pipeline
             output tree. If multiple files match (e.g. stale dirs from
             a previous run, or a stage that appended a suffix to the
             stem), keep the longest one rather than an arbitrary first
             match.

        If nothing is found, logs a specific warning and returns "" so
        callers can fall back to position-based co-occurrence instead
        of failing silently.
        """
        if f.suffix == ".txt":
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    return text
            except Exception:
                pass

        pipeline_root = self.output_dir.parent

        # 2. Original extracted input text
        input_txt = pipeline_root / "input" / f"{stem}.txt"
        if input_txt.exists():
            try:
                text = input_txt.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    return text
            except Exception:
                pass

        # 3. NLP-01 cleaned text, wherever its output dir is actually named
        for nlp01_dir in pipeline_root.glob("nlp_01*"):
            if not nlp01_dir.is_dir():
                continue
            for cleaned in nlp01_dir.rglob("*.txt"):
                if cleaned.stem == stem or cleaned.stem.startswith(stem):
                    try:
                        text = cleaned.read_text(encoding="utf-8", errors="replace")
                        if text.strip():
                            return text
                    except Exception:
                        continue

        # 4. Broad fallback — collect every candidate, keep the longest
        candidates = []
        for txt in pipeline_root.rglob(f"{stem}*.txt"):
            try:
                content = txt.read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    candidates.append(content)
            except Exception:
                continue
        if candidates:
            return max(candidates, key=len)

        self.logger.warning(
            f"  NLP-07: could not locate source text for stem '{stem}' "
            f"anywhere under {pipeline_root} — proximity relations will be "
            "skipped for this file; falling back to position-based "
            "co-occurrence (lower quality, confidence='low' only)."
        )
        return ""

    # ── Relation extraction ───────────────────────────────────────────

    def proximity_relations(self, entities: list, text: str) -> list:
        if not text:
            return []  # nothing to pattern-match against; caller handles fallback

        relations = []
        for i, ent_a in enumerate(entities):
            for ent_b in entities[i + 1:]:
                gap_start = min(ent_a["end"], ent_b["end"])
                gap_end   = max(ent_a["start"], ent_b["start"])
                char_dist = max(0, gap_end - gap_start)
                if char_dist > PROXIMITY_WINDOW:
                    continue

                span_start = max(0, min(ent_a["start"], ent_b["start"]) - 60)
                span_end   = min(len(text), max(ent_a["end"], ent_b["end"]) + 60)
                span_text  = text[span_start:span_end]
                a_label    = ent_a.get("label", "")
                b_label    = ent_b.get("label", "")

                for rel_name, triggers, subj_labels, obj_labels in self.compiled_patterns:
                    matched_strength = None
                    for trigger_pat, strength in triggers:
                        if trigger_pat.search(span_text):
                            matched_strength = strength
                            break
                    if matched_strength is None:
                        continue

                    if a_label in subj_labels and b_label in obj_labels:
                        relations.append({
                            "relation":         rel_name,
                            "subject":          ent_a["text"],
                            "subject_label":    a_label,
                            "object":           ent_b["text"],
                            "object_label":     b_label,
                            "confidence":       score_confidence(matched_strength, char_dist),
                            "trigger_strength": matched_strength,
                            "char_distance":    char_dist,
                            "context":          span_text[:150],
                            "extraction_method": "proximity",
                        })
                        break
                    if b_label in subj_labels and a_label in obj_labels:
                        relations.append({
                            "relation":         rel_name,
                            "subject":          ent_b["text"],
                            "subject_label":    b_label,
                            "object":           ent_a["text"],
                            "object_label":     a_label,
                            "confidence":       score_confidence(matched_strength, char_dist),
                            "trigger_strength": matched_strength,
                            "char_distance":    char_dist,
                            "context":          span_text[:150],
                            "extraction_method": "proximity",
                        })
                        break
        return relations

    def cooccurrence_relations(self, entities: list, text: str) -> list:
        if not text:
            return self._cooccurrence_by_position(entities)

        boundaries = [0]
        for m in re.finditer(r"[।\.!\?]\s+", text):
            boundaries.append(m.end())
        boundaries.append(len(text))

        relations = []
        for idx in range(len(boundaries) - 1):
            s_start   = boundaries[idx]
            s_end     = boundaries[idx + 1]
            sent_ents = [e for e in entities
                         if s_start <= e.get("start", 0) < s_end]
            if len(sent_ents) < 2:
                continue
            for ea, eb in combinations(sent_ents, 2):
                if ea.get("label") != eb.get("label"):
                    relations.append({
                        "relation":         "CO_OCCURS_WITH",
                        "subject":          ea["text"],
                        "subject_label":    ea.get("label", ""),
                        "object":           eb["text"],
                        "object_label":     eb.get("label", ""),
                        "confidence":       "low",
                        "trigger_strength": "low",
                        "char_distance":    abs(ea.get("start",0) - eb.get("start",0)),
                        "context":          text[s_start:s_end][:120],
                        "extraction_method": "cooccurrence",
                    })
        return relations

    def _cooccurrence_by_position(self, entities: list,
                                   window: int = COOCCUR_POSITION_WINDOW) -> list:
        """
        Fallback used only when no source text could be located at all.
        Groups entities that are close together by character offset
        (start/end, as recorded by upstream NER) instead of relying on
        sentence boundaries from `text`. Always confidence='low' — this
        is a last resort, not a substitute for real sentence context.
        """
        relations = []
        sorted_ents = sorted(entities, key=lambda e: e.get("start", 0))
        for i, ea in enumerate(sorted_ents):
            for eb in sorted_ents[i + 1:]:
                gap = eb.get("start", 0) - ea.get("end", 0)
                if gap > window:
                    break  # sorted by start, so nothing further will be closer
                if ea.get("label") != eb.get("label"):
                    relations.append({
                        "relation":         "CO_OCCURS_WITH",
                        "subject":          ea["text"],
                        "subject_label":    ea.get("label", ""),
                        "object":           eb["text"],
                        "object_label":     eb.get("label", ""),
                        "confidence":       "low",
                        "trigger_strength": "low",
                        "char_distance":    max(0, gap),
                        "context":          "",
                        "extraction_method": "cooccurrence_position_fallback",
                    })
        return relations

    def deduplicate(self, relations: list) -> list:
        """Keep highest-confidence relation for each (relation, subject, object) triple."""
        best = {}
        for r in relations:
            key = (r["relation"],
                   r["subject"].strip().lower(),
                   r["object"].strip().lower())
            if CONF_RANK.get(r.get("confidence",""), 0) > CONF_RANK.get(
                    best.get(key, {}).get("confidence",""), 0):
                best[key] = r
        return list(best.values())

    # ── Runner ────────────────────────────────────────────────────────

    def run(self) -> dict:
        output_files          = []
        errors                = []
        all_relations         = []
        global_rel_types:  dict = defaultdict(int)
        global_conf_counts: dict = defaultdict(int)

        for f in self.input_files:
            try:
                stem     = self._get_text_stem(f)
                text     = self._read_text(f, stem)
                entities = self.load_entities(stem)

                if not entities:
                    self.logger.warning(f"  No entities for {f.name} — relations will be empty")

                prox_rels = self.proximity_relations(entities, text)
                cooc_rels = self.cooccurrence_relations(entities, text)

                real_rels = [r for r in prox_rels if r["relation"] != "CO_OCCURS_WITH"]
                combined  = (real_rels + cooc_rels) if real_rels else (prox_rels + cooc_rels)
                final     = self.deduplicate(combined)
                all_relations.extend(final)

                rel_type_summary: dict = defaultdict(int)
                conf_summary:     dict = defaultdict(int)
                for r in final:
                    rel_type_summary[r["relation"]] += 1
                    conf_summary[r["confidence"]]    += 1
                    global_rel_types[r["relation"]]  += 1
                    global_conf_counts[r["confidence"]] += 1

                result = {
                    "source_file":         f.name,
                    "total_relations":     len(final),
                    "relation_types":      dict(rel_type_summary),
                    "confidence_breakdown": dict(conf_summary),
                    "relations":           final,
                }
                out_file = self.output_dir / (stem + "_relations.json")
                out_file.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                output_files.append(out_file)
                self.logger.debug(
                    f"  {f.name}: {len(final)} relations "
                    f"(H:{conf_summary.get('high',0)} "
                    f"M:{conf_summary.get('medium',0)} "
                    f"L:{conf_summary.get('low',0)})"
                )
            except Exception as e:
                errors.append(str(e))
                self.logger.error(f"  ERROR relationship mining for {f.name}: {e}")

        # Global summary
        graph_summary = {
            "total_relations":          len(all_relations),
            "by_type":                  dict(global_rel_types),
            "by_confidence":            dict(global_conf_counts),
            "high_confidence_relations": [r for r in all_relations
                                          if r.get("confidence") == "high"][:100],
        }
        (self.output_dir / "relation_graph_summary.json").write_text(
            json.dumps(graph_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary = (
            f"{len(output_files)} files, {len(all_relations)} relations — "
            f"high:{global_conf_counts.get('high',0)}, "
            f"medium:{global_conf_counts.get('medium',0)}, "
            f"low:{global_conf_counts.get('low',0)} — "
            + ", ".join(f"{k}:{v}" for k, v in global_rel_types.items())
        )
        return {
            "task":                 "NLP-07",
            "output_files":         output_files,
            "total_relations":      len(all_relations),
            "relation_types":       dict(global_rel_types),
            "confidence_breakdown": dict(global_conf_counts),
            "errors":               errors,
            "summary":              summary,
        }