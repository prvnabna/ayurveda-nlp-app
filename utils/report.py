"""
utils/report.py
PipelineReport: collects per-stage results and saves a JSON summary.
"""
import json
from pathlib import Path
from datetime import datetime


class PipelineReport:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.stages = []
        self.started_at = datetime.now().isoformat()

    def add_stage(self, task_id: str, task_name: str, results: dict, elapsed: float):
        self.stages.append({
            "task_id":   task_id,
            "task_name": task_name,
            "elapsed_s": round(elapsed, 2),
            "summary":   results.get("summary", ""),
            "stats":     {k: v for k, v in results.items()
                          if k not in ("output_files", "task")},
        })

    def save(self):
        report = {
            "pipeline": "Ayurveda NLP — PRC Pramana Interns 2026",
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(),
            "stages": self.stages,
        }
        out = self.output_dir / "pipeline_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
