"""
NLP-01: OCR Text Cleaning
=========================
Removes OCR artifacts, invalid characters, duplicate spaces,
encoding issues, and normalises Unicode for Sanskrit/Malayalam/English text.

Deliverable: Python, Regex
"""

import re
import unicodedata
from pathlib import Path


class TextCleaner:
    def __init__(self, input_files, output_dir, logger, lang_hint="auto", prev_outputs=None):
        self.input_files = input_files
        self.output_dir  = Path(output_dir)
        self.logger      = logger
        self.lang_hint   = lang_hint

    # ------------------------------------------------------------------ #
    #  Cleaning rules                                                      #
    # ------------------------------------------------------------------ #

    def fix_encoding(self, text: str) -> str:
        """Fix mojibake and normalise to NFC."""
        text = unicodedata.normalize("NFC", text)
        # Fix common OCR encoding artifacts
        replacements = {
            "\x00": "",    # null bytes
            "\ufffd": "",  # replacement character
            "ﬁ": "fi",    # ligatures
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "\u200b": "",  # zero-width space
            "\u00ad": "",  # soft hyphen
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

    def remove_ocr_artifacts(self, text: str) -> str:
        """Remove common OCR noise patterns."""
        # Remove isolated single non-letter chars that are noise
        text = re.sub(r"(?<!\w)[|\\~`^<>{}]{1,3}(?!\w)", " ", text)
        # Remove repeated punctuation noise (e.g. ........ or ---------)
        text = re.sub(r"([.!?,\-_=*#])\1{3,}", r"\1", text)
        # Remove lone digits surrounded by letters (common OCR 'l' → '1' artifacts)
        text = re.sub(r"(?<=[a-zA-Z])(\d)(?=[a-zA-Z])", r"l", text)
        # Fix split words with hyphen at line break
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # Remove page number patterns
        text = re.sub(r"^\s*[\[\(]?\d{1,4}[\]\)]?\s*$", "", text, flags=re.MULTILINE)
        return text

    def normalise_whitespace(self, text: str) -> str:
        """Collapse multiple spaces/tabs, normalise line endings."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple spaces/tabs into single space
        text = re.sub(r"[ \t]+", " ", text)
        # Keep paragraph breaks (double newline), collapse single newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove trailing whitespace on each line
        text = re.sub(r" +\n", "\n", text)
        return text.strip()

    def clean_devanagari(self, text: str) -> str:
        """Fix common OCR errors in Devanagari (Sanskrit)."""
        # Reorder misplaced matras (common OCR error)
        text = re.sub(r"([\u0902\u0903\u093E-\u094C\u094D])([\u0900-\u097F])",
                      r"\2\1", text)
        # Remove stray combining marks not attached to base
        text = re.sub(r"(?<=[^\u0900-\u097F])[\u0902\u0903\u0941-\u094C]", "", text)
        return text

    def clean_malayalam(self, text: str) -> str:
        """Fix common OCR errors in Malayalam."""
        # Normalize chillu letters
        chillu_map = {
            "\u0D7A": "\u0D28\u0D4D",  # ൺ → ൻ form
            "\u0D7B": "\u0D28\u0D4D",
            "\u0D7C": "\u0D30\u0D4D",
            "\u0D7D": "\u0D32\u0D4D",
            "\u0D7E": "\u0D33\u0D4D",
            "\u0D7F": "\u0D15\u0D4D",
        }
        for ch, repl in chillu_map.items():
            text = text.replace(ch, repl)
        return text

    def detect_language_block(self, text: str):
        """Rough heuristic to decide which extra cleaning to apply."""
        devanagari = len(re.findall(r"[\u0900-\u097F]", text))
        malayalam  = len(re.findall(r"[\u0D00-\u0D7F]", text))
        if devanagari > 10:
            return "sanskrit"
        if malayalam > 10:
            return "malayalam"
        return "english"

    def clean(self, text: str) -> str:
        lang = self.lang_hint if self.lang_hint != "auto" else self.detect_language_block(text)
        text = self.fix_encoding(text)
        text = self.remove_ocr_artifacts(text)
        if lang == "sanskrit":
            text = self.clean_devanagari(text)
        elif lang == "malayalam":
            text = self.clean_malayalam(text)
        text = self.normalise_whitespace(text)
        return text

    # ------------------------------------------------------------------ #
    #  Runner                                                              #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        output_files = []
        stats = {"files": 0, "chars_before": 0, "chars_after": 0, "errors": []}

        for f in self.input_files:
            try:
                raw = f.read_text(encoding="utf-8", errors="replace")
                stats["chars_before"] += len(raw)

                cleaned = self.clean(raw)
                stats["chars_after"] += len(cleaned)

                out_file = self.output_dir / f.name
                out_file.write_text(cleaned, encoding="utf-8")
                output_files.append(out_file)
                stats["files"] += 1
                self.logger.debug(f"  Cleaned: {f.name}")
            except Exception as e:
                stats["errors"].append(str(e))
                self.logger.error(f"  ERROR cleaning {f.name}: {e}")

        reduction = 100 * (1 - stats["chars_after"] / max(stats["chars_before"], 1))
        summary = (f"{stats['files']} files, "
                   f"{stats['chars_before']:,} → {stats['chars_after']:,} chars "
                   f"({reduction:.1f}% reduced)")
        return {
            "task": "NLP-01",
            "output_files": output_files,
            "stats": stats,
            "summary": summary,
        }
