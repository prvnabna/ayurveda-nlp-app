"""
NLP-06: Semantic Tagging — Semantic Assignments for Ayurveda Concepts
======================================================================
Adds semantic tags for Ayurveda concepts and categories.
Maps extracted entities to an ontology/taxonomy.

Deliverable: NLP
"""

import json
import re
from pathlib import Path


# ------------------------------------------------------------------ #
#  Semantic Ontology Mapping                                           #
# ------------------------------------------------------------------ #

SEMANTIC_CATEGORIES = {
    "botanical_medicine": {
        "description": "Herbs and plant-based medicines",
        "entity_labels": ["HERB", "PLANT_PART", "INGREDIENT"],
        "subtypes": {
            "adaptogen":     ["ashwagandha", "brahmi", "shatavari", "guduchi"],
            "digestive":     ["ginger", "pippali", "triphala", "vidanga"],
            "anti_inflammatory": ["turmeric", "neem", "guduchi"],
            "nervine":       ["brahmi", "shankhpushpi", "jatamansi"],
            "immunomodulator": ["tulsi", "amalaki", "guduchi", "ashwagandha"],
        }
    },
    "constitutional_medicine": {
        "description": "Dosha-based constitutional concepts",
        "entity_labels": ["DOSHA", "PROPERTY"],
        "subtypes": {
            "vata_related":  ["vata", "vayu", "space", "air", "movement", "dryness"],
            "pitta_related": ["pitta", "fire", "transformation", "heat", "metabolism"],
            "kapha_related": ["kapha", "earth", "water", "stability", "heaviness"],
        }
    },
    "clinical_entity": {
        "description": "Diseases, conditions, and symptoms",
        "entity_labels": ["DISEASE", "BODY_PART"],
        "subtypes": {
            "metabolic":   ["prameha", "sthoulya", "medoroga", "diabetes"],
            "inflammatory": ["amavata", "shothavyadhi", "arthritis"],
            "digestive":   ["atisara", "grahani", "ajirna", "arsha"],
            "respiratory": ["kasa", "shvasa", "pratishyaya"],
            "skin":        ["kushtha", "visarpa", "dadru"],
        }
    },
    "therapeutic_intervention": {
        "description": "Treatments, therapies, and procedures",
        "entity_labels": ["PROCEDURE"],
        "subtypes": {
            "shodhana":    ["panchakarma", "vamana", "virechana", "basti", "nasya", "raktamokshana"],
            "shamana":     ["rasayana", "langhana", "pachana", "deepana"],
            "external":    ["abhyanga", "shirodhara", "lepa", "avagaha"],
        }
    },
    "classical_reference": {
        "description": "References to classical Ayurveda texts",
        "entity_labels": ["SOURCE_REF"],
        "subtypes": {
            "brihat_trayi": ["charaka samhita", "sushruta samhita", "ashtanga hridayam", "ashtanga sangraha"],
            "laghu_trayi":  ["madhava nidana", "sarangadhara samhita", "bhavaprakasha"],
            "nighantu":     ["dhanvantari nighantu", "raja nighantu", "kaideva nighantu"],
        }
    },
    "pharmacological_property": {
        "description": "Drug properties: Rasa, Guna, Virya, Vipaka",
        "entity_labels": ["PROPERTY"],
        "subtypes": {
            "rasa":   ["madhura", "amla", "lavana", "katu", "tikta", "kashaya"],
            "guna":   ["laghu", "guru", "snigdha", "ruksha", "ushna", "sheeta"],
            "virya":  ["ushna virya", "sheeta virya", "hot potency", "cold potency"],
            "vipaka": ["madhura vipaka", "katu vipaka", "amla vipaka"],
        }
    },
}

# Semantic relation types between entities
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
    def __init__(self, input_files, output_dir, logger, lang_hint="auto", prev_outputs=None):
        self.input_files  = input_files
        self.output_dir   = Path(output_dir)
        self.logger       = logger
        self.lang_hint    = lang_hint
        self.prev_outputs = prev_outputs or {}

        # Build reverse lookup: entity_label → semantic_category
        self.label_to_category = {}
        for cat, info in SEMANTIC_CATEGORIES.items():
            for lbl in info["entity_labels"]:
                self.label_to_category.setdefault(lbl, []).append(cat)

        # Build subtype lookup: keyword → subtype
        self.keyword_to_subtype = {}
        for cat, info in SEMANTIC_CATEGORIES.items():
            for subtype, keywords in info.get("subtypes", {}).items():
                for kw in keywords:
                    self.keyword_to_subtype[kw.lower()] = (cat, subtype)

    def assign_semantic_tag(self, entity: dict) -> dict:
        """Enrich an entity dict with semantic category and subtype."""
        entity = dict(entity)
        text_lower = entity["text"].lower()
        label = entity.get("label", "")

        # Assign category from label
        entity["semantic_categories"] = self.label_to_category.get(label, ["uncategorized"])

        # Assign subtype from keyword match
        for kw, (cat, subtype) in self.keyword_to_subtype.items():
            if kw in text_lower:
                entity["semantic_subtype"] = subtype
                entity["semantic_category_detail"] = cat
                break
        else:
            entity["semantic_subtype"] = None
            entity["semantic_category_detail"] = None

        return entity

    def tag_text_level(self, text: str, entities: list) -> dict:
        """Assign document-level semantic tags based on entity distribution."""
        label_counts = {}
        for e in entities:
            label_counts[e.get("label", "UNK")] = label_counts.get(e.get("label", "UNK"), 0) + 1

        # Determine dominant topic
        topics = []
        if label_counts.get("HERB", 0) > 2:
            topics.append("pharmacognosy")
        if label_counts.get("DISEASE", 0) > 2:
            topics.append("clinical")
        if label_counts.get("PROCEDURE", 0) > 1:
            topics.append("therapeutics")
        if label_counts.get("PROPERTY", 0) > 2:
            topics.append("pharmacology")
        if label_counts.get("SOURCE_REF", 0) > 0:
            topics.append("classical_text_reference")
        if label_counts.get("DOSHA", 0) > 2:
            topics.append("constitutional_medicine")

        return {
            "document_topics": topics or ["general_ayurveda"],
            "entity_distribution": label_counts,
            "richness_score": min(10, len(entities) // 3),  # 0-10 scale
        }

    def load_ner_entities(self, source_stem: str) -> list:
        """Try to load NER results from NLP-05 output."""
        possible_paths = [
            self.output_dir.parent / "nlp_05_entity_recognition_ner_model_training" / (source_stem + "_ner.json"),
            self.output_dir.parent / (source_stem + "_ner.json"),
        ]
        for p in possible_paths:
            if p.exists():
                try:
                    return json.loads(p.read_text())["entities"]
                except Exception:
                    pass
        return []

    def run(self) -> dict:
        output_files = []
        errors = []
        total_tagged = 0

        for f in self.input_files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")

                # Use NLP-05 NER entities if available, otherwise empty
                entities = self.load_ner_entities(f.stem)

                # Enrich each entity with semantic tags
                tagged_entities = [self.assign_semantic_tag(e) for e in entities]
                total_tagged += len(tagged_entities)

                # Document-level tags
                doc_tags = self.tag_text_level(text, tagged_entities)

                result = {
                    "source_file": f.name,
                    "document_semantics": doc_tags,
                    "tagged_entities": tagged_entities,
                    "semantic_schema_version": "1.0",
                }
                out_file = self.output_dir / (f.stem + "_semantic.json")
                out_file.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                output_files.append(f)
                self.logger.debug(f"  Semantic tags for {f.name}: {len(tagged_entities)} entities")
            except Exception as e:
                errors.append(str(e))
                self.logger.error(f"  ERROR semantic tagging {f.name}: {e}")

        # Save the full semantic schema
        schema_path = self.output_dir / "semantic_schema.json"
        schema_path.write_text(
            json.dumps(SEMANTIC_CATEGORIES, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        summary = f"{len(output_files)} files, {total_tagged} entities semantically tagged"
        return {
            "task": "NLP-06",
            "output_files": output_files,
            "total_tagged": total_tagged,
            "summary": summary,
        }
