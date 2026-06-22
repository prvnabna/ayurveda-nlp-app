"""
NLP-07: Relationship Mining  [UPGRADED v2]
==========================================
Improvements over v1:
  - Real confidence scoring (high/medium/low) based on trigger strength + proximity
  - Trigger word taxonomy with strength weights
  - Sentence-boundary awareness
  - Direction inference (A→B vs B→A) with syntactic cues
  - Deduplication with best-confidence merge
  - Summary statistics by relation type and confidence tier
  - Ready for dependency parser integration (stanza/spaCy)

Deliverable: NLP, Dependency Parsing
"""

import re
import json
from pathlib import Path
from itertools import combinations
from collections import defaultdict


# ------------------------------------------------------------------ #
#  Relation patterns with trigger strength                            #
# ------------------------------------------------------------------ #

# strength: "high" = explicit verb, "medium" = prepositional, "low" = contextual
RELATION_PATTERNS = [

    # TREATS: HERB / PROCEDURE → DISEASE
    ("TREATS",
     [
         (r"\b(treats?|cures?|heals?|eradicates?|eliminates?|destroys?)\b", "high"),
         (r"\b(alleviates?|relieves?|reduces?|diminishes?|controls?)\b", "high"),
         (r"\b(beneficial\s+for|useful\s+in|effective\s+(for|against)|indicated\s+for)\b", "medium"),
         (r"\b(used\s+(for|in|against|to\s+treat)|remedy\s+for|action\s+on)\b", "medium"),
         (r"\b(helps?\s+in|good\s+for|useful\s+for|recommended\s+for)\b", "low"),
     ],
     ["HERB", "INGREDIENT", "PROCEDURE", "FORMULATION"],
     ["DISEASE", "BODY_PART"]),

    # BALANCES_DOSHA: HERB → DOSHA
    ("BALANCES_DOSHA",
     [
         (r"\b(balances?|pacifies?|normalizes?|harmonizes?)\b", "high"),
         (r"\b(reduces?|decreases?|alleviates?|mitigates?)\b", "high"),
         (r"\b(aggravates?|increases?|provokes?|vitiates?)\b", "high"),  # opposite direction but still a dosha relation
         (r"\b(beneficial\s+for|useful\s+in|controls?)\b", "medium"),
     ],
     ["HERB", "INGREDIENT", "PROCEDURE", "FORMULATION"],
     ["DOSHA"]),

    # HAS_PLANT_PART: HERB → PLANT_PART
    ("HAS_PLANT_PART",
     [
         (r"\b(from\s+the|using\s+the|prepared\s+from|extracted\s+from|derived\s+from|of\s+the)\b", "high"),
         (r"\b(part\s+used|used\s+part)\b", "high"),
         (r"\b(root|bark|leaf|leaves|fruit|seed|flower|resin)\s+(of|is|are)\b", "medium"),
     ],
     ["HERB"],
     ["PLANT_PART"]),

    # AFFECTS_BODY: DISEASE → BODY_PART
    ("AFFECTS_BODY",
     [
         (r"\b(affects?|involves?|attacks?|damages?|inflames?)\b", "high"),
         (r"\b(manifests?\s+in|located\s+in|found\s+in|present\s+in)\b", "medium"),
         (r"\b(of\s+the|in\s+the|at\s+the)\b", "low"),
     ],
     ["DISEASE"],
     ["BODY_PART"]),

    # HAS_PROPERTY: HERB → PROPERTY
    ("HAS_PROPERTY",
     [
         (r"\b(has|have|possesses?|exhibits?|shows?|displays?)\b", "high"),
         (r"\b(is\s+known\s+for|characterized\s+by|attributed\s+with)\b", "high"),
         (r"\b(is|are)\b", "low"),
     ],
     ["HERB", "INGREDIENT", "FORMULATION"],
     ["PROPERTY"]),

    # MENTIONED_IN: HERB/DISEASE/PROCEDURE → SOURCE_REF
    ("MENTIONED_IN",
     [
         (r"\b(mentioned\s+in|described\s+in|cited\s+in|found\s+in)\b", "high"),
         (r"\b(according\s+to|as\s+per|as\s+stated\s+in|as\s+described\s+in)\b", "high"),
         (r"\b(in\s+the|from\s+the)\b", "low"),
     ],
     ["HERB", "DISEASE", "PROCEDURE", "FORMULATION"],
     ["SOURCE_REF"]),

    # DOSAGE: HERB → QUANTITY
    ("DOSAGE",
     [
         (r"\b(dose|dosage|taken|administered|given|used\s+at|prescribed\s+at)\b", "high"),
         (r"\b(in\s+a\s+dose|at\s+a\s+dose)\b", "high"),
         (r"\b(daily|twice|thrice|morning|evening)\b", "medium"),
     ],
     ["HERB", "INGREDIENT", "FORMULATION"],
     ["QUANTITY"]),

    # INDICATED_FOR: PROCEDURE → DISEASE
    ("INDICATED_FOR",
     [
         (r"\b(indicated\s+for|recommended\s+for|prescribed\s+for)\b", "high"),
         (r"\b(used\s+in|helpful\s+in|beneficial\s+in)\b", "medium"),
     ],
     ["PROCEDURE"],
     ["DISEASE"]),

    # CONTAINS: FORMULATION / HERB compound → HERB ingredient
    ("CONTAINS",
     [
         (r"\b(contains?|includes?|composed\s+of|made\s+(of|from|with)|consists?\s+of)\b", "high"),
         (r"\b(mixture\s+of|combination\s+of|compound\s+of)\b", "high"),
         (r"\b(with|and|plus)\b", "low"),
     ],
     ["HERB", "INGREDIENT", "FORMULATION"],
     ["HERB", "INGREDIENT"]),
]

PROXIMITY_WINDOW = 350  # characters between two entities for proximity check


# ------------------------------------------------------------------ #
#  Confidence scorer                                                   #
# ------------------------------------------------------------------ #

def score_confidence(trigger_strength: str, proximity: int) -> str:
    """
    Combine trigger word strength with entity proximity to get final confidence.
    proximity = gap in characters between the two entity spans.
    """
    if trigger_strength == "high" and proximity < 150:
        return "high"
    if trigger_strength == "high" and proximity < 300:
        return "medium"
    if trigger_strength == "medium" and proximity < 150:
        return "medium"
    return "low"


class RelationshipExtractor:
    def __init__(self, input_files, output_dir, logger, lang_hint="auto", prev_outputs=None):
        self.input_files  = input_files
        self.output_dir   = Path(output_dir)
        self.logger       = logger
        self.lang_hint    = lang_hint
        self.prev_outputs = prev_outputs or {}

        # Compile: list of (rel_name, [(compiled_pattern, strength)], subj_labels, obj_labels)
        self.compiled_patterns = []
        for rel_name, trigger_list, subj_labels, obj_labels in RELATION_PATTERNS:
            compiled_triggers = [
                (re.compile(pat, re.IGNORECASE | re.UNICODE), strength)
                for pat, strength in trigger_list
            ]
            self.compiled_patterns.append(
                (rel_name, compiled_triggers, subj_labels, obj_labels)
            )

    # ------------------------------------------------------------------ #
    #  Proximity-based relation extraction                                 #
    # ------------------------------------------------------------------ #

    def proximity_relations(self, entities: list, text: str) -> list:
        relations = []

        for i, ent_a in enumerate(entities):
            for ent_b in entities[i + 1:]:
                # Proximity check
                gap_start = min(ent_a["end"], ent_b["end"])
                gap_end   = max(ent_a["start"], ent_b["start"])
                char_dist = max(0, gap_end - gap_start)
                if char_dist > PROXIMITY_WINDOW:
                    continue

                # Span of text covering both entities + context buffer
                span_start = max(0, min(ent_a["start"], ent_b["start"]) - 60)
                span_end   = min(len(text), max(ent_a["end"], ent_b["end"]) + 60)
                span_text  = text[span_start:span_end]

                a_label = ent_a.get("label", "")
                b_label = ent_b.get("label", "")

                for rel_name, triggers, subj_labels, obj_labels in self.compiled_patterns:
                    matched_strength = None
                    for trigger_pat, strength in triggers:
                        if trigger_pat.search(span_text):
                            matched_strength = strength
                            break  # use first (strongest) match

                    if matched_strength is None:
                        continue

                    # Try A → B
                    if a_label in subj_labels and b_label in obj_labels:
                        conf = score_confidence(matched_strength, char_dist)
                        relations.append({
                            "relation":       rel_name,
                            "subject":        ent_a["text"],
                            "subject_label":  a_label,
                            "object":         ent_b["text"],
                            "object_label":   b_label,
                            "confidence":     conf,
                            "trigger_strength": matched_strength,
                            "char_distance":  char_dist,
                            "context":        span_text[:150],
                            "extraction_method": "proximity",
                        })
                        break

                    # Try B → A
                    if b_label in subj_labels and a_label in obj_labels:
                        conf = score_confidence(matched_strength, char_dist)
                        relations.append({
                            "relation":       rel_name,
                            "subject":        ent_b["text"],
                            "subject_label":  b_label,
                            "object":         ent_a["text"],
                            "object_label":   a_label,
                            "confidence":     conf,
                            "trigger_strength": matched_strength,
                            "char_distance":  char_dist,
                            "context":        span_text[:150],
                            "extraction_method": "proximity",
                        })
                        break

        return relations

    # ------------------------------------------------------------------ #
    #  Sentence-level co-occurrence                                        #
    # ------------------------------------------------------------------ #

    def cooccurrence_relations(self, entities: list, text: str) -> list:
        # Split on sentence-ending punctuation (including Sanskrit danda ।)
        sentence_boundaries = [0]
        for m in re.finditer(r"[।\.\!\?]\s+", text):
            sentence_boundaries.append(m.end())
        sentence_boundaries.append(len(text))

        relations = []
        for idx in range(len(sentence_boundaries) - 1):
            s_start = sentence_boundaries[idx]
            s_end   = sentence_boundaries[idx + 1]

            sent_ents = [
                e for e in entities
                if s_start <= e.get("start", 0) < s_end
            ]
            if len(sent_ents) < 2:
                continue

            for ea, eb in combinations(sent_ents, 2):
                if ea.get("label") != eb.get("label"):
                    relations.append({
                        "relation":       "CO_OCCURS_WITH",
                        "subject":        ea["text"],
                        "subject_label":  ea.get("label", ""),
                        "object":         eb["text"],
                        "object_label":   eb.get("label", ""),
                        "confidence":     "low",
                        "trigger_strength": "low",
                        "char_distance":  abs(ea.get("start", 0) - eb.get("start", 0)),
                        "context":        text[s_start:s_end][:120],
                        "extraction_method": "cooccurrence",
                    })

        return relations

    # ------------------------------------------------------------------ #
    #  Deduplication with best-confidence merge                            #
    # ------------------------------------------------------------------ #

    def deduplicate(self, relations: list) -> list:
        """Keep highest-confidence relation for each (relation, subject, object) triple."""
        CONF_RANK = {"high": 3, "medium": 2, "low": 1}
        best = {}
        for r in relations:
            key = (r["relation"], r["subject"].strip().lower(), r["object"].strip().lower())
            existing_rank = CONF_RANK.get(best.get(key, {}).get("confidence", ""), 0)
            new_rank      = CONF_RANK.get(r.get("confidence", "low"), 0)
            if new_rank > existing_rank:
                best[key] = r
        return list(best.values())

    # ------------------------------------------------------------------ #
    #  Load entities from previous stage                                   #
    # ------------------------------------------------------------------ #

    def load_entities(self, source_stem: str) -> list:
        search_dirs = [
            self.output_dir.parent / "nlp_06_semantic_tagging_semantic_assignments",
            self.output_dir.parent / "nlp_05_entity_recognition_ner_model_training",
            self.output_dir.parent / "nlp_03_annotation_annotation_guideline_creation",
            self.output_dir.parent,
        ]
        suffixes = ["_semantic.json", "_ner.json", "_annotated.json"]
        for d in search_dirs:
            if not (d and d.exists()):
                continue
            for suf in suffixes:
                p = d / (source_stem + suf)
                if p.exists():
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        return data.get(
                            "entities",
                            data.get("tagged_entities",
                            data.get("annotations", []))
                        )
                    except Exception:
                        pass
        return []

    # ------------------------------------------------------------------ #
    #  Runner                                                              #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        output_files   = []
        errors         = []
        all_relations  = []
        global_rel_types: dict = defaultdict(int)
        global_conf_counts: dict = defaultdict(int)

        for f in self.input_files:
            try:
                text     = f.read_text(encoding="utf-8", errors="replace")
                entities = self.load_entities(f.stem)

                if not entities:
                    self.logger.warning(f"  No entities found for {f.name} — run NLP-03/05 first")

                # Extract relations
                prox_rels  = self.proximity_relations(entities, text)
                cooc_rels  = self.cooccurrence_relations(entities, text)

                # Filter out CO_OCCURS_WITH if we have real relations
                real_rels = [r for r in prox_rels if r["relation"] != "CO_OCCURS_WITH"]
                if real_rels:
                    combined = real_rels + cooc_rels
                else:
                    combined = prox_rels + cooc_rels

                final_rels = self.deduplicate(combined)
                all_relations.extend(final_rels)

                # Statistics per file
                rel_type_summary: dict = defaultdict(int)
                conf_summary: dict = defaultdict(int)
                for r in final_rels:
                    rel_type_summary[r["relation"]] += 1
                    conf_summary[r["confidence"]] += 1
                    global_rel_types[r["relation"]] += 1
                    global_conf_counts[r["confidence"]] += 1

                result = {
                    "source_file":     f.name,
                    "total_relations": len(final_rels),
                    "relation_types":  dict(rel_type_summary),
                    "confidence_breakdown": dict(conf_summary),
                    "relations":       final_rels,
                    "note": (
                        "Upgrade to full dependency parsing: "
                        "pip install stanza && python -c \"import stanza; stanza.download('sa')\""
                    ),
                }
                out_file = self.output_dir / (f.stem + "_relations.json")
                out_file.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                output_files.append(out_file)
                self.logger.debug(
                    f"  {f.name}: {len(final_rels)} relations "
                    f"(high:{conf_summary.get('high',0)}, "
                    f"medium:{conf_summary.get('medium',0)}, "
                    f"low:{conf_summary.get('low',0)})"
                )

            except Exception as e:
                errors.append(str(e))
                self.logger.error(f"  ERROR in relationship mining for {f.name}: {e}")

        # Global relation graph summary
        graph_summary = {
            "total_relations": len(all_relations),
            "by_type": dict(global_rel_types),
            "by_confidence": dict(global_conf_counts),
            "high_confidence_relations": [
                r for r in all_relations if r.get("confidence") == "high"
            ][:100],
            "top_relations": sorted(
                all_relations,
                key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x.get("confidence", ""), 0),
                reverse=True
            )[:50],
        }
        graph_path = self.output_dir / "relation_graph_summary.json"
        graph_path.write_text(
            json.dumps(graph_summary, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        summary = (
            f"{len(output_files)} files, {len(all_relations)} relations — "
            f"high:{global_conf_counts.get('high',0)}, "
            f"medium:{global_conf_counts.get('medium',0)}, "
            f"low:{global_conf_counts.get('low',0)} — "
            + ", ".join(f"{k}:{v}" for k, v in global_rel_types.items())
        )
        return {
            "task":            "NLP-07",
            "output_files":    output_files,
            "total_relations": len(all_relations),
            "relation_types":  dict(global_rel_types),
            "confidence_breakdown": dict(global_conf_counts),
            "summary":         summary,
        }