import pytest  # type: ignore
from rads.datetime64util import (year, month, day, hour, minute, second,
                                 microsecond, ymdhmsus)

import numpy as np


DATE = np.datetime64('2002-02-03T13:56:03.172')
LEAP_YEAR = np.datetime64('2020-02-29T18:25:40.12850')
NEAR_EPOCH = np.datetime64('1970-01-01T00:00:00.080988')
LOW = np.datetime64('0000-01-01T00:00:00.0')
HIGH = np.datetime64('9999-12-31T23:59:59.999999')


def test_year():
    assert year(DATE) == 2002
    assert year(LEAP_YEAR) == 2020
    assert year(NEAR_EPOCH) == 1970
    assert year(LOW) == 0000
    assert year(HIGH) == 9999


def test_month():
    assert month(DATE) == 2
    assert month(LEAP_YEAR) == 2
    assert month(NEAR_EPOCH) == 1
    assert month(LOW) == 1
    assert month(HIGH) == 12


def test_day():
    assert day(DATE) == 3
    assert day(LEAP_YEAR) == 29
    assert day(NEAR_EPOCH) == 1
    assert day(LOW) == 1
    assert day(HIGH) == 31


def test_hour():
    assert hour(DATE) == 13
    assert hour(LEAP_YEAR) == 18
    assert hour(NEAR_EPOCH) == 0
    assert hour(LOW) == 0
    assert hour(HIGH) == 23


def test_minute():
    assert minute(DATE) == 56
    assert minute(LEAP_YEAR) == 25
    assert minute(NEAR_EPOCH) == 0
    assert minute(LOW) == 0
    assert minute(HIGH) == 59


def test_second():
    assert second(DATE) == 3
    assert second(LEAP_YEAR) == 40
    assert second(NEAR_EPOCH) == 0
    assert second(LOW) == 0
    assert second(HIGH) == 59


def test_microsecond():
    assert microsecond(DATE) == 172000
    assert microsecond(LEAP_YEAR) == 128500
    assert microsecond(NEAR_EPOCH) == 80988
    assert microsecond(LOW) == 0
    assert microsecond(HIGH) == 999999


def test_ymdhmsus():
    assert ymdhmsus(DATE) == (2002, 2, 3, 13, 56, 3, 172000)
    assert ymdhmsus(LEAP_YEAR) == (2020, 2, 29, 18, 25, 40, 128500)
    assert ymdhmsus(NEAR_EPOCH) == (1970, 1, 1, 0, 0, 0, 80988)
    assert ymdhmsus(LOW) == (0, 1, 1, 0, 0, 0, 0)
    assert ymdhmsus(HIGH) == (9999, 12, 31, 23, 59, 59, 999999)
