"""
utils/logger.py
===============
Logger for both CLI (run_pipeline.py) and Streamlit (app.py).

FIX: app.py imports get_streamlit_logger — it did not exist, causing
     an ImportError that silently broke the entire pipeline on Streamlit.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


# ── CLI logger (used by run_pipeline.py) ────────────────────────────

def setup_logger(log_dir=None, verbose: bool = False) -> logging.Logger:
    """Return a configured logger that writes to console + optional log file."""
    logger = logging.getLogger("ayurveda_nlp")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s",
                            datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"pipeline_{ts}.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Streamlit logger (used by app.py) ───────────────────────────────

def get_streamlit_logger(log_list: list, verbose: bool = False) -> logging.Logger:
    """
    Return a logger that appends messages to `log_list` (a plain Python list).
    Streamlit reads that list to display the run log in the UI.

    ROOT CAUSE FIX:
    This function was MISSING entirely. app.py does:
        from utils.logger import get_streamlit_logger
    Without it, every pipeline run inside Streamlit threw an ImportError,
    silently falling through to the except block which returned empty results.
    That is why NLP-05, NLP-06, NLP-07 never ran → Relations tab always empty.
    """
    logger_name = f"ayurveda_nlp_st_{id(log_list)}"   # unique per run
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    class _ListHandler(logging.Handler):
        def __init__(self, target: list):
            super().__init__()
            self._list = target

        def emit(self, record: logging.LogRecord):
            try:
                self._list.append(self.format(record))
            except Exception:
                pass

    h = _ListHandler(log_list)
    h.setLevel(logging.DEBUG if verbose else logging.INFO)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)

    # Mirror to stdout for local debugging
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger