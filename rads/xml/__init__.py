"""XML tools for reading the RADS's configuration files."""

try:
    from .lxml import parse
except ImportError:
    from .etree import parse

__all__ = ('parse',)
