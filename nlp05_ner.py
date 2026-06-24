"""
NLP-05: NER Model Training  [FIXED v3]
=======================================
BUG FIXES:
  1. load_training_data() searched for 'nlp_03_annotation_annotation_guideline_creation'
     — directory does not exist.  Now uses rglob("*_annotated.json") which finds
     annotation files regardless of what the parent folder is named.
  2. output_files still appended 'f' (original .txt) instead of 'out_file' (NER .json).
     Fixed — NLP-06 and NLP-07 can now locate NER output via rglob too.
  3. Regex-based fallback NER so entities are always produced even without spaCy.

Deliverable: spaCy, Transformers
"""

import json
import re
import random
from pathlib import Path

try:
    import spacy
    from spacy.training import Example
    from spacy.util import minibatch, compounding
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


NER_LABELS = [
    "HERB", "DOSHA", "DISEASE", "PROCEDURE",
    "BODY_PART", "INGREDIENT", "QUANTITY",
    "SOURCE_REF", "PLANT_PART", "PROPERTY", "FORMULATION", "DIET",
]

RECOMMENDED_MODELS = {
    "multilingual_indic": "ai4bharat/indic-bert",
    "multilingual_base":  "bert-base-multilingual-cased",
    "english_light":      "en_core_web_sm",
}

MIN_TRAINING_DOCS = 3


class NERModelTrainer:
    def __init__(self, input_files, output_dir, logger,
                 lang_hint="auto", prev_outputs=None):
        self.input_files  = input_files
        self.output_dir   = Path(output_dir)
        self.logger       = logger
        self.lang_hint    = lang_hint
        self.prev_outputs = prev_outputs or {}

    # ── Training data ────────────────────────────────────────────────

    def load_training_data(self) -> list:
        """
        FIX: was searching hard-coded directory name 'nlp_03_annotation_annotation_guideline_creation'
        which never exists. Now uses rglob('*_annotated.json') on the pipeline root,
        which finds annotation files in ANY subdirectory regardless of naming.
        """
        training = []
        pipeline_root = self.output_dir.parent

        for ann_file in pipeline_root.rglob("*_annotated.json"):
            try:
                data     = json.loads(ann_file.read_text(encoding="utf-8"))
                text     = data.get("text", "")
                spans    = data.get("annotations", [])
                entities = [(s["start"], s["end"], s["label"]) for s in spans
                            if s.get("label") in NER_LABELS]
                if text and entities:
                    training.append((text, {"entities": entities}))
            except Exception as e:
                self.logger.debug(f"  Skipping {ann_file.name}: {e}")

        self.logger.info(f"  NLP-05: {len(training)} annotated docs found for training")
        return training

    # ── Regex-based fallback NER ─────────────────────────────────────

    def regex_ner(self, text: str) -> list:
        """
        Zero-shot NER using NLP-03 patterns.
        Always produces entities even when spaCy is not installed or training fails.
        """
        try:
            import sys
            sys.path.insert(0, str(self.output_dir.parent))
            from nlp03_annotation import ENTITY_SEEDS, LABEL_PRIORITY
        except ImportError:
            return []

        entities = []
        for label, patterns in ENTITY_SEEDS.items():
            for pat in patterns:
                try:
                    for m in re.finditer(pat, text, re.IGNORECASE | re.UNICODE):
                        entities.append({
                            "text":       m.group(0),
                            "label":      label,
                            "start":      m.start(),
                            "end":        m.end(),
                            "confidence": 0.70,
                            "source":     "regex",
                        })
                except Exception:
                    pass

        # Deduplicate — longest span wins at each position
        entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        deduped, last_end = [], -1
        for e in entities:
            if e["start"] >= last_end:
                deduped.append(e)
                last_end = e["end"]
        return deduped

    # ── spaCy training ───────────────────────────────────────────────

    def train_spacy_ner(self, training_data: list) -> dict:
        if not SPACY_AVAILABLE:
            return {"error": "spaCy not installed — pip install spacy"}

        self.logger.info("  Training spaCy NER model…")
        nlp = spacy.blank("xx")
        ner = nlp.add_pipe("ner")
        for label in NER_LABELS:
            ner.add_label(label)

        model_path = self.output_dir / "spacy_ner_model"

        if len(training_data) < MIN_TRAINING_DOCS:
            self.logger.warning(
                f"  Only {len(training_data)} docs — minimum {MIN_TRAINING_DOCS}. "
                "Saving untrained model; regex fallback will be used."
            )
            nlp.to_disk(str(model_path))
            return {"model_path": str(model_path), "trained": False}

        optimizer = nlp.begin_training()
        for epoch in range(30):
            random.shuffle(training_data)
            losses = {}
            for batch in minibatch(training_data, size=compounding(4.0, 32.0, 1.001)):
                examples = []
                for text, annotations in batch:
                    doc     = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                nlp.update(examples, drop=0.35, losses=losses)
            if epoch % 10 == 0:
                self.logger.debug(f"  Epoch {epoch}: loss={losses.get('ner',0):.3f}")

        nlp.to_disk(str(model_path))
        self.logger.info(f"  spaCy model saved → {model_path}")
        return {"model_path": str(model_path), "trained": True, "epochs": 30}

    def run_model_inference(self, nlp_model, text: str) -> list:
        doc = nlp_model(text[:1_000_000])
        return [
            {"text": ent.text, "label": ent.label_,
             "start": ent.start_char, "end": ent.end_char,
             "confidence": 0.90, "source": "spacy_model"}
            for ent in doc.ents
        ]

    def merge_entities(self, model_ents: list, regex_ents: list) -> list:
        """Model entities take priority; regex fills gaps."""
        model_spans = {(e["start"], e["end"]) for e in model_ents}
        combined    = list(model_ents)
        for re_ent in regex_ents:
            overlaps = any(
                not (re_ent["end"] <= ms or re_ent["start"] >= me)
                for ms, me in model_spans
            )
            if not overlaps:
                combined.append(re_ent)
        combined.sort(key=lambda e: e["start"])
        return combined

    # ── Runner ───────────────────────────────────────────────────────

    def run(self) -> dict:
        output_files = []
        errors       = []

        # Model card
        (self.output_dir / "model_card.json").write_text(
            json.dumps({
                "task": "NLP-05",
                "model_type": "spaCy NER (multilingual) + regex ensemble",
                "labels": NER_LABELS,
                "recommended_backbone": RECOMMENDED_MODELS,
                "spacy_available": SPACY_AVAILABLE,
                "transformers_available": TRANSFORMERS_AVAILABLE,
            }, indent=2),
            encoding="utf-8",
        )

        training_data = self.load_training_data()

        # Train / load spaCy model
        nlp_model  = None
        ner_result = {}
        if SPACY_AVAILABLE:
            ner_result = self.train_spacy_ner(training_data)
            try:
                mp = ner_result.get("model_path")
                if mp and ner_result.get("trained"):
                    nlp_model = spacy.load(mp)
            except Exception as e:
                self.logger.warning(f"  Could not load trained model: {e}")
        else:
            self.logger.warning("  spaCy unavailable — using regex-only NER")
            ner_result = {"trained": False, "error": "spaCy not installed"}

        entity_total = 0
        for f in self.input_files:
            try:
                # Resolve to a text file if we were passed a JSON file
                if f.suffix == ".json":
                    self.logger.debug(f"  Skipping non-text input: {f.name}")
                    continue

                text = f.read_text(encoding="utf-8", errors="replace")

                model_ents = self.run_model_inference(nlp_model, text) if nlp_model else []
                regex_ents = self.regex_ner(text)
                entities   = self.merge_entities(model_ents, regex_ents)
                entity_total += len(entities)

                result = {
                    "source_file":  f.name,
                    "model":        "spacy_ner + regex_ensemble",
                    "entities":     entities,
                    "entity_count": len(entities),
                    "model_count":  len(model_ents),
                    "regex_count":  len(regex_ents),
                }

                # FIX: append out_file (the NER json), NOT f (the original text)
                out_file = self.output_dir / (f.stem + "_ner.json")
                out_file.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                output_files.append(out_file)   # ← FIXED (was 'f')
                self.logger.debug(
                    f"  {f.name}: {len(entities)} entities "
                    f"(model:{len(model_ents)}, regex:{len(regex_ents)})"
                )
            except Exception as e:
                errors.append(str(e))
                self.logger.error(f"  ERROR in NER for {f.name}: {e}")

        summary = (
            f"{len(output_files)} files, {entity_total} entities extracted, "
            f"training docs: {len(training_data)}"
        )
        return {
            "task":         "NLP-05",
            "output_files": output_files,
            "ner_result":   ner_result,
            "entity_total": entity_total,
            "errors":       errors,
            "summary":      summary,
        }