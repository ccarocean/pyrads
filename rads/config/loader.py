import os
from pathlib import Path
from typing import Optional, Iterable

from appdirs import AppDirs, system
from dataclass_builder import MissingFieldError

from ._builders import PreConfigBuilder, SatelliteBuilder
from .ast import ASTEvaluationError
from .grammar import satellite_grammar, pre_config_grammar, dataroot_grammar
from .tree import Config, PreConfig
from .xml_parsers import GlobalParseFailure
from .._typing import PathLike
from ..xml import ParseError, parse

__all__ = ['ConfigError', 'config_files', 'get_dataroot', 'load_config']

_APPNAME = 'pyrads'
_APPDIRS = AppDirs(_APPNAME, appauthor=False, roaming=False)

# TODO: Remove satellites from blacklist.

# GEOS-3 (g3)    - configuration in rads.xml not complete
# SEASAT-A (ss)  - configuration in rads.xml not complete
#
# This is required because unlike the official RADS, PyRADS attempts to
# load all satellites instead of just one and therefore it will break
# for incomplete configurations.
_BLACKLISTED_SATELLITES = ['g3', 'ss']


class ConfigError(Exception):
    """Exception raised when there is a problem loading the configuration file.

    It is usually raised after another more specific exception has been caught.

    Parameters
    ----------
    message
        Error message.
    line
        Line that cause the exception, if known.
    file
        File that caused the exception, if known.
    original
        Optionally the original exception.

    Attributes
    ----------
    message
        Error message.
    line
        Line that cause the exception, if known (None otherwise).
    file
        File that caused the exception, if known (None otherwise).
    original_exception
        Optionally the original exception (None otherwise).

    """
    message: str
    line: Optional[int] = None
    file: Optional[str] = None
    original_exception = Optional[Exception]

    def __init__(self,
                 message: str,
                 line: Optional[int] = None,
                 file: Optional[str] = None,
                 *, original: Optional[Exception] = None) -> None:
        if line is not None:
            self.line = line
        if file:
            self.file = file
        if original:
            self.original_exception = original
        if file or line:
            file = self.file if self.file else ''
            line = self.line if self.line is not None else ''
            super().__init__(f'{file}:{line}: {message}')
        else:
            super().__init__(message)


def _to_config_error(exc: Exception) -> ConfigError:
    """Convert an exception into a :class:`ConfigError`.

    If the exception has file/line information this function will attempt to
    copy it to the new exception.

    Parameters
    ----------
    exc
        The original exception.

    Returns
    -------
    ConfigError
        The new configuration exception.

    """
    if isinstance(exc, ParseError):
        return ConfigError(exc.msg, exc.lineno, exc.filename, original=exc)
    if isinstance(exc, (GlobalParseFailure, ASTEvaluationError)):
        return ConfigError(exc.message, exc.line, exc.file, original=exc)
    return ConfigError(str(exc), original=exc)


def config_files(dataroot_path: Optional[PathLike] = None,
                 rads: bool = True,
                 pyrads: bool = True) -> Iterable[Path]:
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

    Example
    -------
    This can be used to add another xml file when loading the configuration.

    .. code-block:: python

        load_config(config_files=(xml_files() + '/path/to/custom/config.xml'))

    Parameters
    ----------
    dataroot_path
        Optionally specify the path to the RADS dataroot.  If not given
        :func:`get_dataroot` will be used with default arguments to retrieve
        the RADS dataroot.

    rads
        Set to False to disable all RADS only configuration files.  These are
        the configuration files used by the official RADS implementation and
        are (in order):

            1. ``<dataroot>/conf/rads.xml``
            2. ``~/.rads/rads.xml``
            3. ``rads.xml``

    pyrads
        Set to False to disable all PyRADS configuration files.  These are the
        new configuration files added by PyRADS in which PyRADS exclusive tags
        can be used:

            1. ``/etc/pyrads/settings.xml``
            2. ``~/.config/pyrads/settings.xml``
            3. ``pyrads.xml``

        .. note::

            The files listed above are only for Unix, other operating systems
            differ.

    Returns
    -------
    list(pathlib.Path)
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


def get_dataroot(dataroot: Optional[PathLike] = None,
                 *, xml_files: Optional[Iterable[PathLike]] = None) -> Path:
    """Get the RADS dataroot.

    The *dataroot* or RADSDATAROOT as it is referred to in the official RADS
    user manual is a directory that contains the ``conf`` directory along with
    the individual satellite directories full of data.  In particular, it must
    contain ``conf/rads.xml`` to be considered the *dataroot*.

    The priority of retrieving the *dataroot* is as follows:

        1. If the :paramref:`dataroot` is given directly it will be used.
        2. If the ``RADSDATAROOT`` environment variable is set then it will be
           used as the location of the RADS *dataroot*.
        3. If given :paramref:`xml_files` they will be searched for the
           ``<dataroot>`` tag and its value will be used as the location of the
           *dataroot*.
        4. The default PyRADS specific configuration files will be searched for
           the ``<dataroot>`` tag.

    Parameters
    ----------
    dataroot
        Optionally specify the *dataroot* directly, this function will only
        verify that the path is a *dataroot* if this is given.
    xml_files
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

    Returns
    -------
    pathlib.Path
        The path to the RADS *dataroot*.

    Raises
    ------
    RuntimeError
        If the *dataroot* cannot be found or the given/configured *dataroot* is
        not a valid RADS *dataroot*.

    """
    # find the dataroot
    if dataroot is not None:
        dataroot_ = Path(dataroot)
    elif os.getenv('RADSDATAROOT'):
        dataroot_ = Path(os.getenv('RADSDATAROOT')).expanduser()
    else:  # read from XML files
        if xml_files is None:
            config_paths = config_files(dataroot, rads=False, pyrads=True)
        else:
            config_paths = [Path(os.path.expanduser(os.path.expandvars(f)))
                            for f in xml_files]
        config_paths = [p for p in config_paths if p.is_file()]
        env = {}
        for file in config_paths:
            try:
                ast = dataroot_grammar()(parse(file, rootless=True).down())[0]
                ast.eval(env, {})
            except (ParseError, GlobalParseFailure, ASTEvaluationError) as err:
                raise _to_config_error(err) from err
        try:
            dataroot_ = Path(os.path.expanduser(os.path.expandvars(
                env['dataroot'])))
        except KeyError:
            raise RuntimeError('cannot find RADS data directory')

    # verify the dataroot directory
    if dataroot_.is_dir() and (dataroot_ / 'conf' / 'rads.xml').is_file():
        return dataroot_
    raise RuntimeError(
        f"'{str(dataroot_)}' is not a RADS data directory")


def load_config(*,
                dataroot: Optional[PathLike] = None,
                xml_files: Optional[Iterable[PathLike]] = None,
                satellites: Optional[Iterable[str]] = None):
    """Load the PyRADS configuration from one or more XML files.

    Parameters
    ----------
    dataroot
        Optionally set the RADS *dataroot*.  If not given :func:`get_dataroot`
        will be used.  If :paramref:`xml_files` were given they will be passed
        to the :func:`get_dataroot` function.
    xml_files
        Optionally supply the list of XML files to load the configuration from.
        If not given the result of :func:`config_files` will be used as the
        default.
    satellites
        Optionally specify which satellites to load the configuration for by
        their 2 character id strings.  The default is to load the configuration
        for all non-blacklisted satellites.

    Returns
    -------
    Config
        The resulting PyRADS configuration object.

    Raises
    ------
    RuntimeError
        If the *dataroot* cannot be found or the given/configured *dataroot* is
        not a valid RADS *dataroot*.
    ConfigError
        If there is any problem loading the configuration files.

    See Also
    --------
    rads.config.tree.Config
        PyRADS configuration object.

    """
    # load pre-config
    pre_config = _load_preconfig(dataroot=dataroot,
                                 xml_files=xml_files, satellites=satellites)

    # initialize configuration object and satellite configuration builders
    config = Config(pre_config)
    builders = {sat: SatelliteBuilder()
                for sat in pre_config.satellites
                if sat not in pre_config.blacklist}

    # parse and evaluate each configuration file for each satellite
    for file in config.config_files:
        # construct ast
        try:
            ast = satellite_grammar()(parse(file, rootless=True).down())[0]
        except (ParseError, GlobalParseFailure) as err:
            raise _to_config_error(err) from err
        # evaluate ast for each satellite
        for sat, builder in builders.items():
            try:
                ast.eval(builder, {'id': sat})
            except ASTEvaluationError as err:
                raise _to_config_error(err) from err

    # build each satellite configuration object
    for sat, builder in builders.items():
        try:
            config.satellites[sat] = builder.build()
        except MissingFieldError as err:
            raise ConfigError(str(err)) from err

    return config


def _load_preconfig(*, dataroot: Optional[PathLike] = None,
                    xml_files: Optional[Iterable[PathLike]] = None,
                    satellites: Optional[Iterable[str]] = None) -> PreConfig:
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
        try:
            ast = pre_config_grammar()(parse(file, rootless=True).down())[0]
            ast.eval(builder, {})
        except (ParseError, GlobalParseFailure, ASTEvaluationError) as err:
            raise _to_config_error(err) from err
    builder.dataroot = dataroot_
    builder.config_files = list(xml_paths)
    try:
        pre_config = builder.build()
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
        set(pre_config.blacklist).union(set(_BLACKLISTED_SATELLITES)))

    return pre_config


# The paths below are in order of config file loading.


def _rads_xml(dataroot: PathLike) -> Path:
    """Path to the main RADS configuration file.

    This will be at `<dataroot>/conf/rads.xml`.

    .. note:

        PyRADS specific XML tags are not allowed in this file.

    Parameters
    ----------
    dataroot
        Path to the RADS data root.

    Returns
    -------
    Path
        Path to the main RADS configuration file.

    """
    return Path(dataroot) / 'conf' / 'rads.xml'


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

    Returns
    -------
    Path
        Path to the PyRADS site/system wide configuration file.

    """
    if system in ('win32', 'darwin'):
        return Path(_APPDIRS.site_config_dir)
    # appdirs does not handle site config on linux properly
    return Path('/etc') / _APPNAME / 'settings.xml'


def _user_xml() -> Path:
    """Path to the user local RADS configuration file.

    This will be at `~/.rads/rads.xml` regardless of operating system.

    .. note:

        PyRADS specific XML tags are not allowed in this file.

    Returns
    -------
    Path
        Path to the user local RADS configuration file.

    """
    return Path('~/.rads/rads.xml').expanduser()


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

    Returns
    -------
    Path
        Path to the PyRADS user local configuration file.
    """
    return Path(_APPDIRS.user_config_dir) / 'settings.xml'


def _local_xml() -> Path:
    """Path to the local RADS configuration file.

    This will be `rads.xml` in the current directory.

    .. note:

        PyRADS specific XML tags are not allowed in this file.

    Returns
    -------
    Path
        Path to the local RADS configuration file.

    """
    return Path('rads.xml')


def _local_config() -> Path:
    """Path to the local RADS configuration file.

    This will be `pyrads.xml` in the current directory.

    .. note:

        RADS, not only PyRADS overrides are allowed in this file.

    Returns
    -------
    Path
        Path to the local RADS configuration file.

    """
    return Path('pyrads.xml')
