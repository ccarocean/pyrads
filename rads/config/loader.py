"""PyRADS XML file loader functions."""
import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, TypeVar, cast

from appdirs import AppDirs, system  # type: ignore
from dataclass_builder import MissingFieldError

from ..typing import PathLike
from ..xml import ParseError, parse, rads_fixer
from .ast import ASTEvaluationError, NullStatement, Statement
from .builders import PreConfigBuilder, SatelliteBuilder
from .grammar import dataroot_grammar, pre_config_grammar, satellite_grammar
from .tree import Config, PreConfig
from .xml_parsers import Parser, TerminalXMLParseError

__all__ = ["ConfigError", "config_files", "get_dataroot", "load_config", "xml_loader"]

_APPNAME = "pyrads"
_APPDIRS = AppDirs(_APPNAME, appauthor=False, roaming=False)

# TODO: Remove satellites from blacklist.

# GEOS-3 (g3)    - configuration in rads.xml not complete
# SEASAT-A (ss)  - configuration in rads.xml not complete
#
# This is required because unlike the official RADS, PyRADS attempts to
# load all satellites instead of just one and therefore it will break
# for incomplete configurations.
_BLACKLISTED_SATELLITES = ["g3", "ss"]


class ConfigError(Exception):
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


def _to_config_error(exc: Exception) -> ConfigError:
    """Convert an exception into a :class:`rads.config.loader.ConfigError`.

    If the exception has file/line information this function will attempt to
    copy it to the new exception.

    :param exc:
        The original exception.

    :return rads.config.loader.ConfigError:
        The new configuration exception.
    """
    if isinstance(exc, ParseError):
        return ConfigError(exc.msg, exc.lineno, exc.filename, original=exc)
    if isinstance(exc, (TerminalXMLParseError, ASTEvaluationError)):
        return ConfigError(exc.message, exc.line, exc.file, original=exc)
    return ConfigError(str(exc), original=exc)


def config_files(
    dataroot_path: Optional[PathLike] = None, rads: bool = True, pyrads: bool = True
) -> Iterable[Path]:
    """Get a list of the paths that configuration files will be searched for.

    The full list of configuration files for Unix (other operating systems
    differ) are listed below in the order they are returned from this function,
    which is also the default loading order:

        1. <dataroot>/conf/rads.xml
        2. /etc/pyrads/settings.xml
        3. ~/.rads/rads.xml
        4. ~/.config/pyrads/settings.xml
        5. rads.xml
        6. pyrads.xml

    This can be used to add another xml file when loading the configuration.

    .. code-block:: python

        load_config(config_files=(xml_files() + '/path/to/custom/config.xml'))

    :param dataroot_path:
        Optionally specify the path to the RADS dataroot.  If not given
        :func:`get_dataroot` will be used with default arguments to retrieve
        the RADS dataroot.
    :param rads:
        Set to False to disable all RADS only configuration files.  These are
        the configuration files used by the official RADS implementation and
        are (in order):

            1. ``<dataroot>/conf/rads.xml``
            2. ``~/.rads/rads.xml``
            3. ``rads.xml``
    :param pyrads:
        Set to False to disable all PyRADS configuration files.  These are the
        new configuration files added by PyRADS in which PyRADS exclusive tags
        can be used:

            1. ``/etc/pyrads/settings.xml``
            2. ``~/.config/pyrads/settings.xml``
            3. ``pyrads.xml``

        .. note::

            The files listed above are only for Unix, other operating systems
            differ.

    :return:
        The locations that configuration files can be, in the order that they
        are loaded.
    """
    files = []
    if rads:
        files.append(_rads_xml(get_dataroot(dataroot_path)))
    if pyrads:
        files.append(_site_config())
    if rads:
        files.append(_user_xml())
    if pyrads:
        files.append(_user_config())
    if rads:
        files.append(_local_xml())
    if pyrads:
        files.append(_local_config())
    return files


def get_dataroot(
    dataroot: Optional[PathLike] = None,
    *,
    xml_files: Optional[Iterable[PathLike]] = None,
) -> Path:
    """Get the RADS dataroot.

    The *dataroot* or RADSDATAROOT as it is referred to in the official RADS
    user manual is a directory that contains the ``conf`` directory along with
    the individual satellite directories full of data.  In particular, it must
    contain ``conf/rads.xml`` to be considered the *dataroot*.

    The priority of retrieving the *dataroot* is as follows:

        1. If the `dataroot` is given directly it will be used.
        2. If the ``RADSDATAROOT`` environment variable is set then it will be
           used as the location of the RADS *dataroot*.
        3. If given `xml_files` they will be searched for the ``<dataroot>``
           tag and its value will be used as the location of the *dataroot*.
        4. The default PyRADS specific configuration files will be searched for
           the ``<dataroot>`` tag.

    :param dataroot:
        Optionally specify the *dataroot* directly, this function will only
        verify that the path is a *dataroot* if this is given.
    :param xml_files:
        Optionally specify which configuration files to look for the
        ``<dataroot>`` tag in.  This tag specifies the *dataroot* if it exists.
        The files that will be used if this parameter is not given are (for
        Unix, other operating systems will vary):

            1. ``/etc/pyrads/settings.xml``
            2. ``~/.config/pyrads/settings.xml``
            3. ``pyrads.xml``

        .. note::

            The ``<dataroot>`` tag is a PyRADS only tag.  Adding it to any
            official RADS configuration file will result in an error if the
            official RADS implementation is used.

    :return:
        The path to the RADS *dataroot*.

    :raises RuntimeError:
        If the *dataroot* cannot be found or the given/configured *dataroot* is
        not a valid RADS *dataroot*.
    """
    # find the dataroot
    if dataroot is not None:
        dataroot_ = Path(dataroot)
    elif os.getenv("RADSDATAROOT"):
        dataroot_ = Path(os.environ["RADSDATAROOT"]).expanduser()
    else:  # read from XML files
        if xml_files is None:
            config_paths = config_files(dataroot, rads=False, pyrads=True)
        else:
            config_paths = [
                Path(os.path.expanduser(os.path.expandvars(f))) for f in xml_files
            ]
        dataroot_maybe = None
        for file in [p for p in config_paths if p.is_file()]:
            dataroot_maybe = _load_dataroot(file, dataroot)
        if dataroot_maybe is None:
            raise RuntimeError("cannot find RADS data directory")
        dataroot_ = Path(os.path.expanduser(os.path.expandvars(dataroot_maybe)))

    # verify the dataroot directory
    if dataroot_.is_dir() and (dataroot_ / "conf" / "rads.xml").is_file():
        return dataroot_
    raise RuntimeError(f"'{str(dataroot_)}' is not a RADS data directory")


def load_config(
    *,
    dataroot: Optional[PathLike] = None,
    xml_files: Optional[Iterable[PathLike]] = None,
    satellites: Optional[Iterable[str]] = None,
) -> Config:
    """Load the PyRADS configuration from one or more XML files.

    :param dataroot:
        Optionally set the RADS *dataroot*.  If not given :func:`get_dataroot`
        will be used.  If `xml_files` were given they will be passed to the
        :func:`get_dataroot` function.
    :param xml_files:
        Optionally supply the list of XML files to load the configuration from.
        If not given the result of :func:`config_files` will be used as the
        default.
    :param satellites:
        Optionally specify which satellites to load the configuration for by
        their 2 character id strings.  The default is to load the configuration
        for all non-blacklisted satellites.

    :return:
        The resulting PyRADS configuration object.

    :raises RuntimeError:
        If the *dataroot* cannot be found or the given/configured *dataroot* is
        not a valid RADS *dataroot*.
    :raises rads.config.loader.ConfigError:
        If there is any problem loading the configuration files.

    .. seealso::

        :class:`rads.config.tree.Config`
            PyRADS configuration object.
    """
    # load pre-config
    pre_config = _load_preconfig(
        dataroot=dataroot, xml_files=xml_files, satellites=satellites
    )

    # initialize configuration object and satellite configuration builders
    builders: Mapping[str, Any] = {
        sat: SatelliteBuilder()
        for sat in pre_config.satellites
        if sat not in pre_config.blacklist
    }

    # parse and evaluate each configuration file
    for file in pre_config.config_files:
        builders = _load_satellites(file, builders)

    # build each satellite configuration object
    satellites = {}
    for sat, builder in builders.items():
        try:
            satellites[sat] = builder.build()
        except MissingFieldError as err:
            raise ConfigError(str(err)) from err

    return Config(pre_config, satellites)


T = TypeVar("T")


def xml_loader(grammar: Parser) -> Callable[[Callable[..., T]], Callable[..., T]]:
    r"""Decorate a function taking an AST to allow it to take a file path.

    This decorates a function which takes an AST statement (and any extra
    arguments) and replaces the first argument (the AST statement) with a file
    path.  It also adds error handling for loading, parsing, and evaluation (if
    the decorated function attempts to evaluate the AST statement).  It does
    this by converting all errors to :class:`ConfigError`\ s.

    If the file is empty the decorated function will get a
    :class:`rads.config.ast.NullStatement`.

    :param grammar:
        The grammar to parse the file with.

    :return:
        The decorator.

    :raises ConfigError:
        By the decorated function if the given file cannot be loaded, parsed,
        or evaluated.
    """

    def _decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def _loader(source: PathLike, *args: Any, **kwargs: Any) -> T:
            try:
                return func(_load_ast(source, grammar), *args, **kwargs)
            except ASTEvaluationError as err:
                raise _to_config_error(err) from err

        return _loader

    return _decorator


def _load_ast(source: PathLike, grammar: Parser) -> Statement:
    """Load AST from a file.

    :param source:
        Path of the file to load the AST from.
    :param grammar:
        Grammar used to generate the AST.

    :return:
        The resulting AST.  This will be :class:`rads.config.ast.NullStatement`
        if the file is empty.

    :raises ConfigError:
        If there is any problem reading the given XML file or in parsing it
        with the given `grammar`.
    """
    try:
        return cast(Statement, grammar(parse(source, fixer=rads_fixer).down())[0])
    except StopIteration:
        return NullStatement()
    except (ParseError, TerminalXMLParseError) as err:
        raise _to_config_error(err) from err


@xml_loader(dataroot_grammar())
def _load_dataroot(ast: Statement, dataroot: Optional[str] = None) -> Optional[str]:
    env: Dict[str, str] = {}
    ast.eval(env, {})
    return env.get("dataroot", dataroot)


def _load_preconfig(
    *,
    dataroot: Optional[PathLike] = None,
    xml_files: Optional[Iterable[PathLike]] = None,
    satellites: Optional[Iterable[str]] = None,
) -> PreConfig:
    """Load the pre-configuration object.

    See :func:`load_config` for argument documentation.

    """
    # get dataroot and xml paths
    dataroot_ = get_dataroot(dataroot, xml_files=xml_files)
    if xml_files is None:
        xml_files = config_files(dataroot, rads=True, pyrads=True)
    xml_paths = [p for p in [Path(f) for f in xml_files] if p.is_file()]

    # load preconfig
    builder = PreConfigBuilder()
    for file in xml_paths:
        builder = _load_preconfig2(file, builder)
    builder.dataroot = dataroot_
    builder.config_files = list(xml_paths)
    try:
        pre_config: PreConfig = builder.build()
    except MissingFieldError as err:
        raise ConfigError(str(err)) from err

    # allow satellite subsetting
    if satellites is not None:
        # check that all provided satellites are in the configuration files
        for satellite in satellites:
            if satellite not in pre_config.satellites:
                raise ValueError(f"satellite '{satellite}' is not available")
        pre_config.satellites = list(satellites)

    # add hardcoded blacklist
    pre_config.blacklist = list(
        set(pre_config.blacklist).union(set(_BLACKLISTED_SATELLITES))
    )

    return pre_config


@xml_loader(pre_config_grammar())
def _load_preconfig2(ast: Statement, builder: T) -> T:
    ast.eval(builder, {})
    return builder


@xml_loader(satellite_grammar())
def _load_satellites(ast: Statement, builders: Mapping[str, T]) -> Mapping[str, T]:
    for sat, builder in builders.items():
        ast.eval(builder, {"id": sat})
    return builders


# The paths below are in order of config file loading.


def _rads_xml(dataroot: PathLike) -> Path:
    """Path to the main RADS configuration file.

    This will be at `<dataroot>/conf/rads.xml`.

    .. note:

        PyRADS specific XML tags are not allowed in this file.

    :param dataroot:
        Path to the RADS data root.

    :return:
        Path to the main RADS configuration file.
    """
    return Path(dataroot) / "conf" / "rads.xml"


def _site_config() -> Path:
    r"""Path to the PyRADS site/system configuration file.

    ================  ================================================
    Operating System  Path
    ================  ================================================
    Mac OS X          /Library/Application Support/pyrads/settings.xml
    Unix              /etc/pyrads
    Windows           C:\ProgramData\pyrads\settings.xml
    ================  ================================================

    .. note:

        RADS, not only PyRADS overrides are allowed in this file.

    :return:
        Path to the PyRADS site/system wide configuration file.
    """
    if system in ("win32", "darwin"):
        return Path(_APPDIRS.site_config_dir)
    # appdirs does not handle site config on linux properly
    return Path("/etc") / _APPNAME / "settings.xml"


def _user_xml() -> Path:
    """Path to the user local RADS configuration file.

    This will be at `~/.rads/rads.xml` regardless of operating system.

    .. note:

        PyRADS specific XML tags are not allowed in this file.

    :return:
        Path to the user local RADS configuration file.
    """
    return Path("~/.rads/rads.xml").expanduser()


def _user_config() -> Path:
    r"""Path to the PyRADS user local configuration file.

    ================  =====================================================
    Operating System  Path
    ================  =====================================================
    Mac OS X          ~/Library/Preferences/pyrads/settings.xml
    Unix              ~/.config/pyrads/settings.xml
    Windows           C:\Users\<username>\AppData\Local\pyrads\settings.xml
    ================  =====================================================

    .. note:

        RADS, not only PyRADS overrides are allowed in this file.

    :return:
        Path to the PyRADS user local configuration file.
    """
    return Path(_APPDIRS.user_config_dir) / "settings.xml"


def _local_xml() -> Path:
    """Path to the local RADS configuration file.

    This will be `rads.xml` in the current directory.

    .. note:

        PyRADS specific XML tags are not allowed in this file.

    :return:
        Path to the local RADS configuration file.
    """
    return Path("rads.xml")


def _local_config() -> Path:
    """Path to the local RADS configuration file.

    This will be `pyrads.xml` in the current directory.

    .. note:

        RADS, not only PyRADS overrides are allowed in this file.

    :return:
        Path to the local RADS configuration file.
    """
    return Path("pyrads.xml")
