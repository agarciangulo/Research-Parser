import logging
import os


def setup_logger(name: str) -> logging.Logger:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, level, logging.INFO))
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
