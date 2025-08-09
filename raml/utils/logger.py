import logging
import sys


def setup_logger(name, log_file=None, level=logging.INFO):
    """Configure a logger with console and optional file output.

    Args:
        name: Logger name, typically __name__ from the calling module.
        log_file: Optional path to a log file.
        level: Logging level.

    Returns:
        Configured logging.Logger instance.
    """
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file is not None:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
