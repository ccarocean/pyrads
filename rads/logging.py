import logging
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING
from typing import Union, Optional
from .typing import PathLike

__all__ = [
    "configure_logging",
    "log",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]

log = logging.getLogger(__name__.split(".")[0])


def configure_logging(level: Union[str, int], file: Optional[PathLike] = None):
    for handler in log.handlers:
        log.removeHandler(handler)

    if file:
        handler = logging.FileHandler(str(file))
    else:
        handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%d-%b-%y %H:%M:%S"
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(level)


# default logging level
configure_logging(WARNING)
