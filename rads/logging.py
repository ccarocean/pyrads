import logging
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING
from typing import Union

__all__ = [
    "configure_logging",
    "logger",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]

logger = logging.getLogger(__name__.split(".")[0])


def configure_logging(level: Union[str, int]):
    for handler in logger.handlers:
        logger.removeHandler(handler)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%b-%y %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


# default logging level
configure_logging(WARNING)
