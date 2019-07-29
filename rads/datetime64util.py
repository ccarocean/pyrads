"""Additional utility for numpy.datetime64."""

from typing import Tuple, overload

import numpy as np  # type: ignore


def year(datetime64: np.datetime64) -> np.generic:
    """Get year from NumPy datetime64 value/array.

    Parameters
    ----------
    datetime64
        Value/array to get year number(s) from.

    Returns
    -------
    np.int or nd.array[np.int]
        Year or array of years from :paramref:`datetime64`.

    """
    # based on: https://stackoverflow.com/a/26895491
    return datetime64.astype("datetime64[Y]").astype(int) + 1970


def month(datetime64: np.datetime64) -> np.generic:
    """Get month from NumPy datetime64 value/array.

    Parameters
    ----------
    datetime64
        Value/array to get month number(s) from.

    Returns
    -------
    np.int or nd.array[np.int]
        Month or array of months from :paramref:`datetime64`.

    """
    # based on: https://stackoverflow.com/a/26895491
    return datetime64.astype("datetime64[M]").astype(int) % 12 + 1


def day(datetime64: np.datetime64) -> np.generic:
    """Get day of month from NumPy datetime64 value/array.

    Parameters
    ----------
    datetime64
        Value/array to get day(s) of month from.

    Returns
    -------
    np.int or nd.array[np.int]
        Day of month or array of days of month from :paramref:`datetime64`.

    """
    # based on: https://stackoverflow.com/a/26895491
    return (
        datetime64.astype("datetime64[D]") - datetime64.astype("datetime64[M]")
    ).astype(int) + 1


def hour(datetime64: np.datetime64) -> np.generic:
    """Get hour from NumPy datetime64 value/array.

    Parameters
    ----------
    datetime64
        Value/array to get hour(s) from.

    Returns
    -------
    np.int or nd.array[np.int]
        Hour or array of hours from :paramref:`datetime64`.

    """
    return datetime64.astype("datetime64[h]").astype(int) % 24


def minute(datetime64: np.datetime64) -> np.generic:
    """Get minute from NumPy datetime64 value/array.

    Parameters
    ----------
    datetime64
        Value/array to get minute(s) from.

    Returns
    -------
    np.int or nd.array[np.int]
        Minute or array of minutes from :paramref:`datetime64`.

    """
    return datetime64.astype("datetime64[m]").astype(int) % 60


def second(datetime64: np.datetime64) -> np.generic:
    """Get second from NumPy datetime64 value/array.

    Parameters
    ----------
    datetime64
        Value/array to get second(s) from.

    Returns
    -------
    np.int or nd.array[np.int]
        Second or array of seconds from :paramref:`datetime64`.

    """
    return datetime64.astype("datetime64[s]").astype(int) % 60


def microsecond(datetime64: np.datetime64) -> np.generic:
    """Get microsecond from NumPy datetime64 value/array.

    Parameters
    ----------
    datetime64
        Value/array to get microsecond(s) from.

    Returns
    -------
    np.int or nd.array[np.int]
        Microsecond or array of microseconds from :paramref:`datetime64`.

    """
    return datetime64.astype("datetime64[us]").astype(int) % 1000000


def ymdhmsus(
    datetime64: np.datetime64
) -> Tuple[
    np.generic, np.generic, np.generic, np.generic, np.generic, np.generic, np.generic
]:
    """Get time components from NumPy datetime64 value/array.

    Parameters
    ----------
    datetime64
        Value/array to get time components from.

    Returns
    -------
    np.int or nd.array[np.int]
        Year or array of years from :paramref:`datetime64`.
    np.int or nd.array[np.int]
        Month or array of months from :paramref:`datetime64`.
    np.int or nd.array[np.int]
        Day of month or array of days of month from :paramref:`datetime64`.
    np.int or nd.array[np.int]
        Hour or array of hours from :paramref:`datetime64`.
    np.int or nd.array[np.int]
        Minute or array of minutes from :paramref:`datetime64`.
    np.int or nd.array[np.int]
        Second or array of seconds from :paramref:`datetime64`.
    np.int or nd.array[np.int]
        Microsecond or array of microseconds from :paramref:`datetime64`.

    """
    # pylint: disable=redefined-outer-name
    # based on: https://stackoverflow.com/a/26895491
    year = datetime64.astype("datetime64[Y]").astype(int) + 1970
    month_ = datetime64.astype("datetime64[M]")
    month = month_.astype(int) % 12 + 1
    day = (datetime64.astype("datetime64[D]") - month_).astype(int) + 1
    hour = datetime64.astype("datetime64[h]").astype(int) % 24
    minute = datetime64.astype("datetime64[m]").astype(int) % 60
    second = datetime64.astype("datetime64[s]").astype(int) % 60
    microsecond = datetime64.astype("datetime64[us]").astype(int) % 1000000
    return year, month, day, hour, minute, second, microsecond
