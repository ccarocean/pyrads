from datetime import datetime

import numpy as np
import pytest
from cf_units import Unit

from rads.config.text_parsers import (TerminalTextParseError, TextParseError,
                                      lift, list_of, range_of, one_of,
                                      compress, cycles, nop, ref_pass,
                                      repeat, time, unit)
from rads.config.tree import Range, Compress, Cycles, ReferencePass, Repeat


def test_exceptions():
    # A TextParseError is also a TerminalTextParseError because exceptions for
    # TerminalTextParseErrors should also catch TextParseErrors.
    with pytest.raises(TerminalTextParseError):
        raise TextParseError


def test_lift():
    assert lift(str)('3', {}) == '3'
    assert lift(int)('3', {}) == 3
    assert lift(float)('3.14', {}) == 3.14
    with pytest.raises(TextParseError):
        lift(float)('not_a_float', {})
    with pytest.raises(TerminalTextParseError):
        lift(float, terminal=True)('not_a_float', {})
    assert lift(int)._lifted == 'int'


def test_list_of(mocker):
    assert list_of(lift(int))('1 2 3 4 5', {}) == [1, 2, 3, 4, 5]
    assert list_of(lift(int))('1   2 3      4 5', {}) == [1, 2, 3, 4, 5]
    assert list_of(lift(int))('', {}) == []
    m = mocker.Mock()
    m.return_value = 3
    list_of(m)('abc 123', {'stuff': 3.14})
    assert m.call_count == 2
    m.assert_any_call('abc', {'stuff': 3.14})
    m.assert_any_call('123', {'stuff': 3.14})
    with pytest.raises(TextParseError):
        list_of(lift(int))('a b c', {})
    with pytest.raises(TerminalTextParseError):
        list_of(lift(int), terminal=True)('a b c', {})


def test_range_of(mocker):
    assert range_of(lift(int))('2 3', {}) == Range(2, 3)
    assert range_of(lift(float))('2.5 3.14', {}) == Range(2.5, 3.14)
    assert range_of(lift(int))('2   3', {}) == Range(2, 3)
    m = mocker.Mock()
    range_of(m)('2.5 3.14', {'stuff': 3.14})
    assert m.call_count == 2
    m.assert_any_call('2.5', {'stuff': 3.14})
    m.assert_any_call('3.14', {'stuff': 3.14})
    with pytest.raises(TextParseError):
        range_of(lift(int))('2.5 3.14', {})
    with pytest.raises(TextParseError):
        range_of(lift(int))(' ', {})
    with pytest.raises(TextParseError):
        range_of(lift(int))('1', {})
    with pytest.raises(TextParseError):
        range_of(lift(int))('1 2 3', {})
    with pytest.raises(TerminalTextParseError):
        range_of(lift(int), terminal=True)(' ', {})


def test_one_of(mocker):
    int_float_str = one_of((lift(int), lift(float), lift(str)))
    assert int_float_str('3', {}) == 3
    assert type(int_float_str('3', {})) == int
    assert int_float_str('3.14', {}) == 3.14
    assert type(int_float_str('3.14', {})) == float
    assert int_float_str('3abc', {}) == '3abc'
    assert type(int_float_str('3abc', {})) == str
    m1 = mocker.Mock()
    m2 = mocker.Mock()
    one_of((m1, m2))('abc', {'stuff': 123})
    m1.assert_called_once_with('abc', {'stuff': 123})
    assert m2.call_count == 0
    with pytest.raises(TextParseError):
        one_of((lift(int), lift(float)))('abc', {})
    with pytest.raises(TextParseError):
        one_of((time,))('abc', {})
    with pytest.raises(TerminalTextParseError):
        one_of((time,), terminal=True)('abc', {})


def test_compress():
    assert compress('int1', {}) == Compress(np.int8)
    assert compress('int2', {}) == Compress(np.int16)
    assert compress('int4', {}) == Compress(np.int32)
    assert compress('real', {}) == Compress(np.float32)
    assert compress('dble', {}) == Compress(np.float64)
    assert compress('int4 2e-3', {}) == Compress(np.int32, 2e-3)
    assert compress('int4 2d-3', {}) == Compress(np.int32, 2e-3)
    assert compress('int4 2e-3 1300e3', {}) == Compress(np.int32, 2e-3, 1300e3)
    assert compress('int4 2d-3 1300d3', {}) == Compress(np.int32, 2e-3, 1300e3)
    assert compress('int4 2-3 1300+3', {}) == Compress(np.int32, 2e-3, 1300e3)
    assert (compress('int4   2d-3    1300d3', {}) ==
            Compress(np.int32, 2e-3, 1300e3))
    with pytest.raises(TextParseError):
        compress('', {})
    with pytest.raises(TextParseError):
        compress('int', {})
    with pytest.raises(TextParseError):
        compress('int2 2a-3', {})
    with pytest.raises(TextParseError):
        compress('int2 2e-3 1300a3', {})
    with pytest.raises(TextParseError):
        compress('int4 2e-3 1300e3 5', {})


def test_cycles():
    assert cycles('100 200', {}) == Cycles(100, 200)
    assert cycles('  100    200       ', {}) == Cycles(100, 200)
    with pytest.raises(TextParseError):
        cycles('100 2.5', {})
    with pytest.raises(TextParseError):
        cycles('1.5 200', {})
    with pytest.raises(TextParseError):
        cycles('', {})
    with pytest.raises(TextParseError):
        cycles('100', {})
    with pytest.raises(TextParseError):
        cycles('100 200 300', {})


def test_nop():
    assert nop('  abc   ', {}) == '  abc   '


def test_ref_pass():
    assert (ref_pass('2016-03-01T21:00:11 -41.5 116 105', {}) ==
            ReferencePass(datetime(2016, 3, 1, 21, 0, 11), -41.5, 116, 105))
    assert (ref_pass('2016-03-01T21:00:11 -41.5 116 105 3', {}) ==
            ReferencePass(datetime(2016, 3, 1, 21, 0, 11), -41.5, 116, 105, 3))
    # invalid values
    with pytest.raises(TextParseError):
        ref_pass('2016-03-01T99:00:11 -41.5 116 105 3', {})
    with pytest.raises(TextParseError):
        ref_pass('2016-03-01T21:00:11 2016-03-01T21:00:11 116 105 3', {})
    with pytest.raises(TextParseError):
        ref_pass('2016-03-01T21:00:11 -41.5 116.5 105 3', {})
    with pytest.raises(TextParseError):
        ref_pass('2016-03-01T21:00:11 -41.5 116 105.5 3', {})
    with pytest.raises(TextParseError):
        ref_pass('2016-03-01T21:00:11 -41.5 116 105 3.5', {})
    # missing values
    with pytest.raises(TextParseError):
        ref_pass('', {})
    with pytest.raises(TextParseError):
        ref_pass('2016-03-01T21:00:11', {})
    with pytest.raises(TextParseError):
        ref_pass('2016-03-01T21:00:11 -41.5', {})
    with pytest.raises(TextParseError):
        ref_pass('2016-03-01T21:00:11 -41.5 116', {})
    # too many values
    with pytest.raises(TextParseError):
        ref_pass('2016-03-01T21:00:11 -41.5 116 105 3 5', {})


def test_repeat():
    assert repeat('9.91 254', {}) == Repeat(9.91, 254)
    assert repeat('9.91 254 0.2', {}) == Repeat(9.91, 254, 0.2)
    # invalid values
    with pytest.raises(TextParseError):
        repeat('abc 254', {})
    with pytest.raises(TextParseError):
        repeat('9.91 254.5', {})
    with pytest.raises(TextParseError):
        repeat('9.91 254 abc', {})
    # missing values
    with pytest.raises(TextParseError):
        repeat('', {})
    with pytest.raises(TextParseError):
        repeat('9.91', {})
    # too many values
    with pytest.raises(TextParseError):
        repeat('9.91 254 0.2 1', {})


def test_time():
    assert time('2012-05-07T14:57:08', {}) == datetime(2012, 5, 7, 14, 57, 8)
    assert time('2012-05-07T14:57', {}) == datetime(2012, 5, 7, 14, 57)
    assert time('2012-05-07T14', {}) == datetime(2012, 5, 7, 14)
    assert time('2012-05-07T', {}) == datetime(2012, 5, 7)
    assert time('2012-05-07', {}) == datetime(2012, 5, 7)
    with pytest.raises(TextParseError):
        time('2012-05', {})
    with pytest.raises(TextParseError):
        time('2012-05-07 14:57:08', {})


def test_unit():
    assert unit('km', {}) == Unit('km')
    assert unit('dB', {}) == Unit('no unit')
    assert unit('decibel', {}) == Unit('no unit')
    assert unit('yymmddhhmmss', {}) == Unit('unknown')
    with pytest.raises(TextParseError):
        unit('abc', {})
