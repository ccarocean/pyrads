"""XML tools for reading the RADS's configuration files."""

try:
    from .lxml import etree
except ImportError:
    from .etree import etree
