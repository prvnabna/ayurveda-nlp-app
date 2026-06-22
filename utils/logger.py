"""
utils/logger.py
Simple logger setup for the Ayurveda NLP Pipeline.
"""
import logging
from pathlib import Path
from datetime import datetime


def setup_logger(log_dir: Path, verbose: bool = False) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("ayurveda_nlp")
    logger.setLevel(level)
    return logger


class StreamlitHandler(logging.Handler):
    """Logging handler that appends to a list (for Streamlit display)."""
    def __init__(self, log_list: list):
        super().__init__()
        self.log_list = log_list

    def emit(self, record):
        self.log_list.append(self.format(record))


def get_streamlit_logger(log_list: list, verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger(f"ayurveda_nlp_st_{id(log_list)}")
    logger.setLevel(level)
    logger.handlers = []
    handler = StreamlitHandler(log_list)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger
