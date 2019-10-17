"""Log instance and logging configuration."""

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
"""The PyRADS logger, use :func:`configure_logging` to set level and destination."""


def configure_logging(level: Union[str, int], file: Optional[PathLike] = None) -> None:
    """Configure logging for PyRADS.

    This will automatically be called upon import with the following arguments
    but can be called again to change logging.

    .. code-block:: python

        configure_logging(WARNING)

    :param level:
        The logging level, can be any of:
            * :data:`DEBUG`
            * :data:`INFO`
            * :data:`WARNING` (PyRADS default)
            * :data:`ERROR`
            * :data:`CRITICAL`
    :param file:
        Optionally specify a filename to write the log to.  By default all log
        messages will be written to *stderr*.
    """
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
