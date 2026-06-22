"""
NLP-02: Multi-language Detection
=================================
Detects Sanskrit, Malayalam, and English content automatically.
Segments mixed documents by language block.

Deliverable: NLP, Indic NLP
"""

import re
import json
from pathlib import Path
from collections import Counter


# Unicode ranges for each script
SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),   # Sanskrit / Hindi
    "malayalam":  (0x0D00, 0x0D7F),
    "latin":      (0x0041, 0x024F),    # English + extended
    "arabic":     (0x0600, 0x06FF),
    "tamil":      (0x0B80, 0x0BFF),
    "telugu":     (0x0C00, 0x0C7F),
}

AYURVEDA_TERMS_SANSKRIT = {
    "आयुर्वेद", "दोष", "वात", "पित्त", "कफ", "धातु", "रस", "औषध",
    "चरक", "सुश्रुत", "अष्टांग", "त्रिदोष", "पञ्चकर्म",
}
AYURVEDA_TERMS_MALAYALAM = {
    "ആയുർവേദം", "ദോഷം", "വാതം", "പിത്തം", "കഫം", "ഔഷധം", "ചരകൻ",
    "ചികിത്സ", "സസ്യം", "ഗ്രന്ഥം",
}
AYURVEDA_TERMS_ENGLISH = {
    "ayurveda", "dosha", "vata", "pitta", "kapha", "herb", "treatment",
    "manuscript", "sloka", "panchakarma", "rasayana",
}


class LanguageDetector:
    def __init__(self, input_files, output_dir, logger, lang_hint="auto", prev_outputs=None):
        self.input_files = input_files
        self.output_dir  = Path(output_dir)
        self.logger      = logger
        self.lang_hint   = lang_hint

    # ------------------------------------------------------------------ #
    #  Detection logic                                                     #
    # ------------------------------------------------------------------ #

    def char_counts(self, text: str) -> dict:
        counts = {k: 0 for k in SCRIPT_RANGES}
        for ch in text:
            cp = ord(ch)
            for script, (lo, hi) in SCRIPT_RANGES.items():
                if lo <= cp <= hi:
                    counts[script] += 1
                    break
        return counts

    def detect_language(self, text: str) -> dict:
        """Return language breakdown as percentage dict."""
        cc = self.char_counts(text)
        total = sum(cc.values()) or 1

        lang_map = {
            "sanskrit": cc["devanagari"],
            "malayalam": cc["malayalam"],
            "english":  cc["latin"],
        }
        total_lang = sum(lang_map.values()) or 1

        percentages = {k: round(100 * v / total_lang, 1) for k, v in lang_map.items()}
        dominant = max(percentages, key=percentages.get)

        # Check for Ayurveda term signals
        text_lower = text.lower()
        signals = {
            "ayurveda_en": sum(1 for t in AYURVEDA_TERMS_ENGLISH if t in text_lower),
            "ayurveda_sa": sum(1 for t in AYURVEDA_TERMS_SANSKRIT if t in text),
            "ayurveda_ml": sum(1 for t in AYURVEDA_TERMS_MALAYALAM if t in text),
        }

        mixed = sum(1 for p in percentages.values() if p > 15) > 1

        return {
            "dominant":    dominant,
            "is_mixed":    mixed,
            "percentages": percentages,
            "ayurveda_signals": signals,
            "char_counts": cc,
        }

    def segment_by_language(self, text: str) -> list:
        """Split text into language-tagged segments (paragraph level)."""
        paragraphs = re.split(r"\n{2,}", text)
        segments = []
        for para in paragraphs:
            if not para.strip():
                continue
            info = self.detect_language(para)
            segments.append({
                "text":     para.strip(),
                "language": info["dominant"],
                "mixed":    info["is_mixed"],
                "scores":   info["percentages"],
            })
        return segments

    # ------------------------------------------------------------------ #
    #  Runner                                                              #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        output_files = []
        lang_summary = Counter()
        errors = []

        for f in self.input_files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                detection = self.detect_language(text)
                segments  = self.segment_by_language(text)

                result = {
                    "source_file": f.name,
                    "detection":   detection,
                    "segments":    segments,
                }

                out_file = self.output_dir / (f.stem + "_lang.json")
                out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
                output_files.append(f)   # Pass original (cleaned) files forward
                lang_summary[detection["dominant"]] += 1

                self.logger.debug(f"  {f.name}: {detection['dominant']} "
                                  f"({detection['percentages']})")
            except Exception as e:
                errors.append(str(e))
                self.logger.error(f"  ERROR detecting language in {f.name}: {e}")

        summary = (f"{sum(lang_summary.values())} files — "
                   + ", ".join(f"{k}: {v}" for k, v in lang_summary.items()))
        return {
            "task": "NLP-02",
            "output_files": output_files,
            "lang_summary": dict(lang_summary),
            "summary": summary,
        }
