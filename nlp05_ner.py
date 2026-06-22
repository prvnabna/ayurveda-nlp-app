"""
NLP-05: NER Model Training — Named Entity Recognition for Ayurveda
===================================================================
Trains or runs a NER model for Ayurveda entities.
Uses spaCy (fast, CPU) with optional HuggingFace transformer backbone.

Deliverable: spaCy, Transformers
"""

import json
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
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        pipeline as hf_pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Labels matching NLP-03 annotation schema
NER_LABELS = [
    "HERB", "DOSHA", "DISEASE", "PROCEDURE",
    "BODY_PART", "INGREDIENT", "QUANTITY",
    "SOURCE_REF", "PLANT_PART", "PROPERTY",
]

# Multilingual models suitable for Sanskrit/Malayalam/English
RECOMMENDED_MODELS = {
    "multilingual": "ai4bharat/indic-bert",       # Best for Indic langs
    "english":      "en_core_web_sm",              # spaCy English
    "lightweight":  "bert-base-multilingual-cased" # Fallback
}


class NERModelTrainer:
    def __init__(self, input_files, output_dir, logger, lang_hint="auto", prev_outputs=None):
        self.input_files  = input_files
        self.output_dir   = Path(output_dir)
        self.logger       = logger
        self.lang_hint    = lang_hint
        self.prev_outputs = prev_outputs or {}

    # ------------------------------------------------------------------ #
    #  Data loading from NLP-03 annotations                               #
    # ------------------------------------------------------------------ #

    def load_training_data(self):
        """Load annotated spans from NLP-03 output."""
        training = []

        # Look for NLP-03 output in prev stage
        nlp03_dir = None
        for task_num, result in self.prev_outputs.items():
            if result.get("task") == "NLP-03":
                # output_files from NLP-03 are the cleaned text files,
                # annotation JSONs are in the NLP-03 output dir
                if result.get("output_files"):
                    sample = result["output_files"][0]
                    nlp03_dir = sample.parent.parent / "nlp_03_annotation_annotation_guideline_creation"
                break

        # Try to find annotation JSONs
        ann_dirs = [nlp03_dir] if nlp03_dir else []
        ann_dirs.append(self.output_dir.parent)  # also search sibling dirs

        for search_dir in ann_dirs:
            if search_dir and search_dir.exists():
                for ann_file in search_dir.glob("*_annotated.json"):
                    try:
                        data = json.loads(ann_file.read_text(encoding="utf-8"))
                        text = data.get("text", "")
                        spans = data.get("annotations", [])
                        entities = [(s["start"], s["end"], s["label"]) for s in spans]
                        if text and entities:
                            training.append((text, {"entities": entities}))
                    except Exception:
                        pass

        return training

    # ------------------------------------------------------------------ #
    #  spaCy NER                                                           #
    # ------------------------------------------------------------------ #

    def train_spacy_ner(self, training_data: list) -> dict:
        """Create and train a blank spaCy NER model."""
        if not SPACY_AVAILABLE:
            return {"error": "spaCy not installed. Run: pip install spacy"}

        self.logger.info("  Training spaCy NER model...")
        nlp = spacy.blank("xx")  # multilingual blank model
        ner = nlp.add_pipe("ner")
        for label in NER_LABELS:
            ner.add_label(label)

        if not training_data:
            self.logger.warning("  No training data found — saving untrained model")
            model_path = self.output_dir / "spacy_ner_model"
            nlp.to_disk(str(model_path))
            return {"model_path": str(model_path), "trained": False}

        optimizer = nlp.begin_training()
        for epoch in range(30):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    examples.append(example)
                nlp.update(examples, drop=0.35, losses=losses)
            if epoch % 10 == 0:
                self.logger.info(f"  Epoch {epoch}: NER loss = {losses.get('ner', 0):.3f}")

        model_path = self.output_dir / "spacy_ner_model"
        nlp.to_disk(str(model_path))
        self.logger.info(f"  spaCy model saved → {model_path}")
        return {"model_path": str(model_path), "trained": True, "epochs": 30}

    # ------------------------------------------------------------------ #
    #  Run NER inference on files                                          #
    # ------------------------------------------------------------------ #

    def run_ner_inference(self, nlp_model, text: str) -> list:
        """Run NER on text and return entity list."""
        doc = nlp_model(text[:1_000_000])  # safety limit
        return [
            {"text": ent.text, "label": ent.label_,
             "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

    # ------------------------------------------------------------------ #
    #  Runner                                                              #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        output_files = []
        errors = []

        # Write model card
        model_card = {
            "task": "NLP-05",
            "model_type": "spaCy NER (multilingual blank)",
            "labels": NER_LABELS,
            "recommended_backbone": RECOMMENDED_MODELS,
            "training_notes": (
                "For best Sanskrit/Malayalam results, fine-tune "
                "ai4bharat/indic-bert with spaCy-transformers. "
                "Install: pip install spacy spacy-transformers transformers"
            ),
            "spacy_available": SPACY_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
        }
        card_path = self.output_dir / "model_card.json"
        card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")

        # Load training data from NLP-03 output
        training_data = self.load_training_data()
        self.logger.info(f"  NLP-05: {len(training_data)} annotated docs loaded for NER training")

        # Train / load model
        ner_result = {}
        nlp_model = None
        if SPACY_AVAILABLE:
            ner_result = self.train_spacy_ner(training_data)
            try:
                model_path = ner_result.get("model_path")
                if model_path:
                    nlp_model = spacy.load(model_path)
            except Exception:
                pass
        else:
            self.logger.warning("  spaCy not available — skipping model training")
            ner_result = {"error": "spaCy not installed"}

        # Run inference on each file and save NER output
        entity_total = 0
        for f in self.input_files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                entities = []
                if nlp_model:
                    entities = self.run_ner_inference(nlp_model, text)
                    entity_total += len(entities)

                result = {
                    "source_file": f.name,
                    "model": "spacy_ner_ayurveda",
                    "entities": entities,
                    "entity_count": len(entities),
                }
                out_file = self.output_dir / (f.stem + "_ner.json")
                out_file.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                output_files.append(f)
            except Exception as e:
                errors.append(str(e))
                self.logger.error(f"  ERROR in NER for {f.name}: {e}")

        summary = (f"{len(output_files)} files processed, "
                   f"{entity_total} entities extracted, "
                   f"training docs: {len(training_data)}")
        return {
            "task": "NLP-05",
            "output_files": output_files,
            "ner_result": ner_result,
            "entity_total": entity_total,
            "summary": summary,
        }
