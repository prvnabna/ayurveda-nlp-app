"""
utils/report.py — Pipeline report generator
"""
import json
from pathlib import Path
from datetime import datetime


class PipelineReport:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.stages: list = []
        self.start_time = datetime.now()

    def add_stage(self, task_id: str, task_name: str, results: dict, elapsed: float):
        self.stages.append({
            "task_id":   task_id,
            "task_name": task_name,
            "elapsed_s": round(elapsed, 2),
            "summary":   results.get("summary", ""),
            "files_out": len(results.get("output_files", [])),
        })

    def save(self) -> Path:
        total = round((datetime.now() - self.start_time).total_seconds(), 2)
        report = {
            "pipeline":      "Ayurveda NLP Pipeline — PRC Pramana Interns 2026",
            "run_date":      self.start_time.isoformat(),
            "total_elapsed": total,
            "stages":        self.stages,
        }
        path = self.output_dir / "pipeline_report.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return path