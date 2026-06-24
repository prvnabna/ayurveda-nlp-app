"""
NLP-06: Semantic Tagging  [FIXED v3]
======================================
BUG FIX:
  load_ner_entities() searched for hard-coded directory
  'nlp_05_entity_recognition_ner_model_training' which does not exist.
  Now uses rglob(stem + '_ner.json') to find NER files anywhere in the
  pipeline output tree — works regardless of directory naming convention.

Also falls back to _annotated.json if no NER file found.

Deliverable: NLP
"""

import json
import re
from pathlib import Path


SEMANTIC_CATEGORIES = {
    "botanical_medicine": {
        "description": "Herbs and plant-based medicines",
        "entity_labels": ["HERB", "PLANT_PART", "INGREDIENT", "FORMULATION"],
        "subtypes": {
            "adaptogen":         ["ashwagandha", "brahmi", "shatavari", "guduchi"],
            "digestive":         ["ginger", "pippali", "triphala", "vidanga", "chitrak"],
            "anti_inflammatory": ["turmeric", "neem", "guduchi", "shallaki"],
            "nervine":           ["brahmi", "shankhpushpi", "jatamansi", "vacha"],
            "immunomodulator":   ["tulsi", "amalaki", "guduchi", "ashwagandha"],
            "hepatoprotective":  ["bhringraj", "kutki", "kalmegh", "punarnava"],
            "diuretic":          ["gokshura", "varuna", "punarnava"],
            "respiratory":       ["vasa", "kantakari", "tulsi", "pippali"],
        },
    },
    "constitutional_medicine": {
        "description": "Dosha-based constitutional concepts",
        "entity_labels": ["DOSHA", "PROPERTY"],
        "subtypes": {
            "vata_related":  ["vata", "vayu", "apana", "prana", "samana", "udana", "vyana"],
            "pitta_related": ["pitta", "pachaka", "ranjaka", "sadhaka", "alochaka", "bhrajaka"],
            "kapha_related": ["kapha", "avalambaka", "kledaka", "bodhaka", "tarpaka", "shleshaka"],
        },
    },
    "clinical_entity": {
        "description": "Diseases, conditions, and symptoms",
        "entity_labels": ["DISEASE", "BODY_PART"],
        "subtypes": {
            "metabolic":    ["prameha", "madhumeha", "sthoulya", "medoroga", "diabetes"],
            "inflammatory": ["amavata", "vatarakta", "shotha", "arthritis"],
            "digestive":    ["atisara", "grahani", "ajirna", "arsha", "gulma"],
            "respiratory":  ["kasa", "shvasa", "pratishyaya"],
            "skin":         ["kushtha", "visarpa", "dadru", "shvitra"],
            "neurological": ["vatavyadhi", "gridhrasi", "apasmara", "kampavata"],
            "urinary":      ["mutrakriccha", "ashmari", "prameha"],
        },
    },
    "therapeutic_intervention": {
        "description": "Treatments, therapies, and procedures",
        "entity_labels": ["PROCEDURE"],
        "subtypes": {
            "shodhana":  ["panchakarma", "vamana", "virechana", "basti", "nasya", "raktamokshana"],
            "shamana":   ["rasayana", "langhana", "pachana", "deepana"],
            "external":  ["abhyanga", "shirodhara", "lepa", "avagaha", "pizhichil", "navarakizhi"],
            "surgical":  ["shastrakarma", "agnikarma", "ksharakarma"],
        },
    },
    "classical_reference": {
        "description": "References to classical Ayurveda texts",
        "entity_labels": ["SOURCE_REF"],
        "subtypes": {
            "brihat_trayi": ["charaka", "sushruta", "ashtanga hridayam", "ashtanga sangraha"],
            "laghu_trayi":  ["madhava nidana", "sarangadhara", "bhavaprakasha"],
            "nighantu":     ["dhanvantari nighantu", "raja nighantu", "kaideva nighantu"],
        },
    },
    "pharmacological_property": {
        "description": "Drug properties: Rasa, Guna, Virya, Vipaka",
        "entity_labels": ["PROPERTY"],
        "subtypes": {
            "rasa":   ["madhura", "amla", "lavana", "katu", "tikta", "kashaya"],
            "guna":   ["laghu", "guru", "snigdha", "ruksha", "ushna", "sheeta",
                       "tikshna", "manda", "mridu", "sthira"],
            "virya":  ["ushna virya", "sheeta virya"],
            "vipaka": ["madhura vipaka", "katu vipaka", "amla vipaka"],
        },
    },
    "formulation": {
        "description": "Classical Ayurveda compound formulations",
        "entity_labels": ["FORMULATION"],
        "subtypes": {
            "churna":   ["triphala churna", "trikatu churna", "ashwagandha churna"],
            "ghrita":   ["brahmi ghrita", "panchagavya ghrita"],
            "taila":    ["mahanarayan taila", "dhanwantaram taila", "kshirabala taila"],
            "arishta":  ["saraswatarishta", "ashwagandharishta", "draksharishta"],
            "vati":     ["chandraprabha vati", "arogyavardhini vati"],
            "guggul":   ["kanchanar guggul", "yogaraja guggul", "triphala guggul"],
        },
    },
    "diet_lifestyle": {
        "description": "Dietary and lifestyle recommendations",
        "entity_labels": ["DIET"],
        "subtypes": {
            "pathya":   ["pathya", "wholesome", "recommended diet"],
            "apathya":  ["apathya", "unwholesome", "contraindicated"],
        },
    },
}

RELATION_SEMANTICS = {
    "HERB_TREATS_DISEASE":   ("HERB",      "DISEASE"),
    "HERB_BALANCES_DOSHA":   ("HERB",      "DOSHA"),
    "PROCEDURE_FOR_DISEASE": ("PROCEDURE", "DISEASE"),
    "HERB_PART_USED":        ("HERB",      "PLANT_PART"),
    "DISEASE_AFFECTS_BODY":  ("DISEASE",   "BODY_PART"),
    "PROPERTY_OF_HERB":      ("PROPERTY",  "HERB"),
    "CITED_IN_SOURCE":       ("HERB",      "SOURCE_REF"),
}


class SemanticTagger:
    def __init__(self, input_files, output_dir, logger,
                 lang_hint="auto", prev_outputs=None):
        self.input_files  = input_files
        self.output_dir   = Path(output_dir)
        self.logger       = logger
        self.lang_hint    = lang_hint
        self.prev_outputs = prev_outputs or {}

        # Build label → category lookup
        self.label_to_category = {}
        for cat, info in SEMANTIC_CATEGORIES.items():
            for lbl in info["entity_labels"]:
                self.label_to_category.setdefault(lbl, []).append(cat)

        # Build keyword → subtype lookup
        self.keyword_to_subtype = {}
        for cat, info in SEMANTIC_CATEGORIES.items():
            for subtype, keywords in info.get("subtypes", {}).items():
                for kw in keywords:
                    self.keyword_to_subtype[kw.lower()] = (cat, subtype)

    # ── Load NER entities ────────────────────────────────────────────

    def load_ner_entities(self, source_stem: str) -> list:
        """
        FIX: was searching 'nlp_05_entity_recognition_ner_model_training' (wrong).
        Now uses rglob to find *_ner.json or *_annotated.json anywhere in the
        pipeline output, so it always works regardless of folder naming.
        """
        pipeline_root = self.output_dir.parent

        # Priority: NER output > annotated output
        for suffix in ["_ner.json", "_annotated.json"]:
            for p in pipeline_root.rglob(source_stem + suffix):
                try:
                    data     = json.loads(p.read_text(encoding="utf-8"))
                    entities = data.get("entities",
                               data.get("tagged_entities",
                               data.get("annotations", [])))
                    if entities:
                        self.logger.debug(
                            f"  NLP-06: loaded {len(entities)} entities "
                            f"from {p.relative_to(pipeline_root)}"
                        )
                        return entities
                except Exception as e:
                    self.logger.debug(f"  Could not read {p}: {e}")

        self.logger.warning(f"  NLP-06: no entities found for '{source_stem}' — "
                            "check NLP-03 / NLP-05 ran successfully")
        return []

    # ── Tagging ──────────────────────────────────────────────────────

    def assign_semantic_tag(self, entity: dict) -> dict:
        entity     = dict(entity)
        text_lower = entity.get("text", "").lower()
        label      = entity.get("label", "")

        entity["semantic_categories"] = self.label_to_category.get(label, ["uncategorized"])

        for kw, (cat, subtype) in self.keyword_to_subtype.items():
            if kw in text_lower:
                entity["semantic_subtype"]          = subtype
                entity["semantic_category_detail"]  = cat
                break
        else:
            entity["semantic_subtype"]         = None
            entity["semantic_category_detail"] = None

        return entity

    def tag_text_level(self, entities: list) -> dict:
        label_counts = {}
        for e in entities:
            lbl = e.get("label", "UNK")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        topics = []
        if label_counts.get("HERB", 0) > 2:       topics.append("pharmacognosy")
        if label_counts.get("DISEASE", 0) > 2:    topics.append("clinical")
        if label_counts.get("PROCEDURE", 0) > 1:  topics.append("therapeutics")
        if label_counts.get("PROPERTY", 0) > 2:   topics.append("pharmacology")
        if label_counts.get("SOURCE_REF", 0) > 0: topics.append("classical_text_reference")
        if label_counts.get("DOSHA", 0) > 2:      topics.append("constitutional_medicine")
        if label_counts.get("FORMULATION", 0) > 0: topics.append("formulation")

        return {
            "document_topics":    topics or ["general_ayurveda"],
            "entity_distribution": label_counts,
            "richness_score":     min(10, len(entities) // 3),
        }

    # ── Runner ───────────────────────────────────────────────────────

    def run(self) -> dict:
        output_files = []
        errors       = []
        total_tagged = 0

        for f in self.input_files:
            try:
                # Resolve stem — input may be a .json (NER output) or .txt
                stem = f.stem
                if stem.endswith("_ner"):
                    stem = stem[:-4]
                elif stem.endswith("_annotated"):
                    stem = stem[:-10]

                # Read source text (best effort)
                text = ""
                if f.suffix == ".txt":
                    text = f.read_text(encoding="utf-8", errors="replace")
                else:
                    # Try to find the original .txt in pipeline root
                    for txt in self.output_dir.parent.rglob(stem + ".txt"):
                        text = txt.read_text(encoding="utf-8", errors="replace")
                        break

                entities        = self.load_ner_entities(stem)
                tagged_entities = [self.assign_semantic_tag(e) for e in entities]
                total_tagged   += len(tagged_entities)
                doc_tags        = self.tag_text_level(tagged_entities)

                result = {
                    "source_file":        f.name,
                    "document_semantics": doc_tags,
                    "tagged_entities":    tagged_entities,
                    "semantic_schema_version": "2.0",
                }
                out_file = self.output_dir / (stem + "_semantic.json")
                out_file.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                output_files.append(out_file)
                self.logger.debug(
                    f"  {f.name}: {len(tagged_entities)} entities semantically tagged"
                )
            except Exception as e:
                errors.append(str(e))
                self.logger.error(f"  ERROR semantic tagging {f.name}: {e}")

        # Save full schema
        (self.output_dir / "semantic_schema.json").write_text(
            json.dumps(SEMANTIC_CATEGORIES, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary = f"{len(output_files)} files, {total_tagged} entities semantically tagged"
        return {
            "task":         "NLP-06",
            "output_files": output_files,
            "total_tagged": total_tagged,
            "errors":       errors,
            "summary":      summary,
        }