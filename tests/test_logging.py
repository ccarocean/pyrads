from rads.logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, configure_logging, log


def log_and_read(level, tmp_path):
    logfile = tmp_path / "logfile.log"
    configure_logging(level, logfile)
    log.debug("a debug message")
    log.info("an info message")
    log.warning("a warning message")
    log.error("an error message")
    log.critical("a critical message")
    return [line[19:] for line in logfile.read_text().splitlines()]


def test_logging_to_file_at_level_debug(tmp_path):
    assert log_and_read(DEBUG, tmp_path) == [
        "DEBUG: a debug message",
        "INFO: an info message",
        "WARNING: a warning message",
        "ERROR: an error message",
        "CRITICAL: a critical message",
    ]


def test_logging_to_file_at_level_info(tmp_path):
    assert log_and_read(INFO, tmp_path) == [
        "INFO: an info message",
        "WARNING: a warning message",
        "ERROR: an error message",
        "CRITICAL: a critical message",
    ]


def test_logging_to_file_at_level_warning(tmp_path):
    assert log_and_read(WARNING, tmp_path) == [
        "WARNING: a warning message",
        "ERROR: an error message",
        "CRITICAL: a critical message",
    ]


def test_logging_to_file_at_level_error(tmp_path):
    assert log_and_read(ERROR, tmp_path) == [
        "ERROR: an error message",
        "CRITICAL: a critical message",
    ]


def test_logging_to_file_at_level_critical(tmp_path):
    assert log_and_read(CRITICAL, tmp_path) == ["CRITICAL: a critical message"]
