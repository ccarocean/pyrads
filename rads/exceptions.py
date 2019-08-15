"""Public exceptions."""

from typing import Optional

__all__ = ["RADSError", "ConfigError", "InvalidDataroot"]


class RADSError(Exception):
    """Base class for all public PyRADS exceptions."""


class ConfigError(RADSError):
    """Exception raised when there is a problem loading the configuration file.

    It is usually raised after another more specific exception has been caught.
    """

    message: str
    """Error message."""
    line: Optional[int] = None
    """Line that cause the exception, if known (None otherwise)."""
    file: Optional[str] = None
    """File that caused the exception, if known (None otherwise)."""
    original_exception: Optional[Exception] = None
    """Optionally the original exception (None otherwise)."""

    def __init__(
        self,
        message: str,
        line: Optional[int] = None,
        file: Optional[str] = None,
        *,
        original: Optional[Exception] = None,
    ):
        """
        :param message:
            Error message.
        :param line:
            Line that cause the exception, if known.
        :param file:
            File that caused the exception, if known.
        :param original:
            Optionally the original exception.
        """
        if line is not None:
            self.line = line
        if file:
            self.file = file
        if original is not None:
            self.original_exception = original
        if file or line:
            file_ = self.file if self.file else ""
            line_ = self.line if self.line is not None else ""
            super().__init__(f"{file_}:{line_}: {message}")
        else:
            super().__init__(message)


class InvalidDataroot(RADSError):
    """Raised when the RADS dataroot is missing or invalid."""
