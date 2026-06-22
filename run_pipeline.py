"""
Ayurveda NLP Pipeline — PRC Pramana Interns 2026  [UPGRADED v2]
================================================================
Runs NLP-01 through NLP-08 on OCR output files automatically.
NLP-08 is new: consolidation + quality scoring for Harini & Neha.

Usage:
    python run_pipeline.py --input <folder_with_ocr_txt_files>
    python run_pipeline.py --input ./input --tasks 1 2 3
    python run_pipeline.py --input ./input --file doc1.txt
    python run_pipeline.py --input ./input --tasks 8   # only consolidation
"""

import argparse
import sys
import json
import time
import logging
from pathlib import Path

from nlp01_text_cleaning    import TextCleaner
from nlp02_language_detect  import LanguageDetector
from nlp03_annotation       import AnnotationGuidelineApplier
from nlp05_ner              import NERModelTrainer
from nlp06_semantic_tagging import SemanticTagger
from nlp07_relationship     import RelationshipExtractor
from nlp08_consolidation    import NLPConsolidator
from utils.logger           import setup_logger
from utils.report           import PipelineReport

TASKS = {
    1: ("NLP-01", "Text Cleaning",         TextCleaner),
    2: ("NLP-02", "Language Detection",    LanguageDetector),
    3: ("NLP-03", "Annotation",            AnnotationGuidelineApplier),
    5: ("NLP-05", "Entity Recognition",    NERModelTrainer),
    6: ("NLP-06", "Semantic Tagging",      SemanticTagger),
    7: ("NLP-07", "Relationship Mining",   RelationshipExtractor),
    8: ("NLP-08", "Consolidation",         NLPConsolidator),   # NEW
}

PIPELINE_ORDER = [1, 2, 3, 5, 6, 7, 8]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ayurveda OCR → NLP Pipeline v2 (NLP-01 to NLP-08)"
    )
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--output", "-o", default="./output")
    parser.add_argument("--tasks",  "-t", nargs="*", type=int, default=PIPELINE_ORDER)
    parser.add_argument("--file",   "-f", default=None)
    parser.add_argument("--lang",   "-l", default="auto",
                        choices=["auto", "sanskrit", "malayalam", "english", "mixed"])
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def collect_files(input_path: Path, single_file=None):
    if single_file:
        f = Path(single_file) if Path(single_file).is_absolute() else input_path / single_file
        if not f.exists():
            print(f"[ERROR] File not found: {f}")
            sys.exit(1)
        return [f]
    files = sorted(input_path.glob("*.txt"))
    if not files:
        print(f"[WARNING] No .txt files found in {input_path}")
        sys.exit(1)
    return files


def run_pipeline(args):
    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_path / "logs", verbose=args.verbose)
    logger.info("=" * 60)
    logger.info("  Ayurveda NLP Pipeline v2 — Starting")
    logger.info(f"  Input : {input_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Tasks : {args.tasks}")
    logger.info("=" * 60)

    files = collect_files(input_path, getattr(args, "file", None))
    logger.info(f"  Files found: {len(files)}")

    report         = PipelineReport(output_path)
    task_sequence  = [t for t in PIPELINE_ORDER if t in args.tasks]
    current_inputs = files
    stage_outputs  = {}

    for task_num in task_sequence:
        task_id, task_name, TaskClass = TASKS[task_num]
        logger.info(f"\n{'─'*60}")
        logger.info(f"  Running {task_id}: {task_name}")
        logger.info(f"{'─'*60}")

        task_out_dir = output_path / f"{task_id.lower().replace('-','_')}_{task_name.lower().replace(' ','_')}"
        task_out_dir.mkdir(parents=True, exist_ok=True)

        task = TaskClass(
            input_files=current_inputs,
            output_dir=task_out_dir,
            logger=logger,
            lang_hint=args.lang,
            prev_outputs=stage_outputs,
        )
        t0 = time.time()
        results = task.run()
        elapsed = time.time() - t0

        stage_outputs[task_num] = results
        report.add_stage(task_id, task_name, results, elapsed)

        if results.get("output_files"):
            current_inputs = results["output_files"]

        logger.info(f"  ✓ {task_id} done in {elapsed:.1f}s — {results.get('summary', '')}")

    report.save()

    logger.info(f"\n{'='*60}")
    logger.info(f"  Pipeline complete!")
    logger.info(f"{'='*60}")

    print(f"\n✅ All done. Outputs in: {output_path}/")
    print(f"📄 Report     : {output_path}/pipeline_report.json")

    qa_dir = output_path / "qa_exports"
    if qa_dir.exists():
        print(f"\n📤 QA Exports (hand these to QA team):")
        print(f"   → {qa_dir}/ner_output.xlsx         (Harini — NER Validator tab)")
        print(f"   → {qa_dir}/relations_output.csv    (Neha   — Relationship tab)")

    nlp08_dir = output_path / "nlp_08_consolidation"
    if nlp08_dir.exists():
        print(f"\n📊 NLP-08 Consolidated Outputs:")
        print(f"   → {nlp08_dir}/ner_consolidated.xlsx")
        print(f"   → {nlp08_dir}/relations_consolidated.csv")
        print(f"   → {nlp08_dir}/nlp08_health.json  ← pipeline quality report")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)