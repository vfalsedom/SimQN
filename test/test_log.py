"""Compatibility checks for the Loguru-backed logging entry points."""

import io
import logging
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger

import qns.utils as utils
import qns.utils.log as log
from qns.simulator import Simulator, func_to_event


@pytest.fixture
def messages(monkeypatch):
    # Other simulator tests configure this shared logger during collection.
    monkeypatch.setattr(log.logger, "handlers", [log.handle])
    monkeypatch.setattr(log.logger, "filters", [])
    monkeypatch.setattr(log.logger, "disabled", False)
    monkeypatch.setattr(log.logger, "propagate", False)
    monkeypatch.setattr(log.handle, "level", logging.NOTSET)
    monkeypatch.setattr(log.handle, "formatter", None)
    monkeypatch.setattr(log.handle, "filters", [])
    monkeypatch.delattr(log.logger, "_simulator", raising=False)
    previous_level = log.logger.level
    log.logger.setLevel(logging.DEBUG)
    captured = []
    sink = loguru_logger.add(captured.append, format=log.format_record, level=0,
                             backtrace=False, diagnose=False, catch=False)
    try:
        yield captured
    finally:
        loguru_logger.remove(sink)
        log.logger.setLevel(previous_level)


@pytest.mark.parametrize("method,level", [
    ("debug", "DEBUG"), ("info", "INFO"), ("warn", "WARNING"),
    ("error", "ERROR"), ("critical", "CRITICAL"),
])
@pytest.mark.parametrize("with_simulator", [False, True])
def test_helpers_keep_levels_formatting_and_time(messages, method, level, with_simulator):
    if with_simulator:
        log.install(SimpleNamespace(tc=1.25))
    getattr(log, method)("value=%d, literal={unchanged}", 7)
    assert len(messages) == 1
    record = messages[0].record
    prefix = "[1.25] " if with_simulator else ""
    assert record["message"] == prefix + "value=7, literal={unchanged}"
    assert record["level"].name == level
    assert record["file"].path == str(Path(__file__).resolve())
    assert record["function"] == "test_helpers_keep_levels_formatting_and_time"


def test_existing_exports_and_logger_identity():
    assert log.logger is logging.getLogger("qns")
    for name in ("logger", "install", "configure", "debug", "info", "warn", "error", "critical", "monitor"):
        assert getattr(utils, name) is getattr(log, name)


def test_set_level_and_lazy_formatting(messages):
    class Unformattable:
        def __str__(self):
            raise AssertionError("Disabled logs must not format their arguments")

    log.logger.setLevel("ERROR")
    assert log.logger.getEffectiveLevel() == logging.ERROR
    assert not log.logger.isEnabledFor(logging.INFO)
    log.info("%s", Unformattable())
    log.logger.error("%(value)s", {"value": 42})
    assert [m.record["message"] for m in messages] == ["42"]


def test_logger_methods_extra_filters_and_handler_level(messages):
    class RejectHidden(logging.Filter):
        def filter(self, record):
            return record.getMessage() != "hidden"

    reject = RejectHidden()
    log.logger.addFilter(reject)
    log.logger.warning("hidden")
    log.logger.removeFilter(reject)
    log.handle.setLevel(logging.WARNING)
    log.logger.info("below handler level")
    log.logger.log(35, "visible %s", "message", extra={"run_id": 3})
    assert len(messages) == 1
    assert messages[0].record["level"].no == 35
    original = messages[0].record["extra"]["logging_record"]
    assert original.name == "qns"
    assert original.run_id == 3
    assert original.getMessage() == "visible message"


def test_monitor_keeps_columns_and_avoids_automatic_time_prefix(messages):
    log.monitor("a", 2)
    log.install(SimpleNamespace(tc=2.5))
    log.monitor("a", 2, sep=";", with_time=True)
    log.monitor("a", 2)
    log.logger.info("direct")
    assert [m.record["message"] for m in messages] == [
        "a,2", "2.5;a;2", "a,2", "direct",
    ]


def test_standard_handlers_can_replace_default_handler(messages):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    log.logger.removeHandler(log.handle)
    log.logger.addHandler(handler)
    try:
        log.info("value=%d", 7)
        assert stream.getvalue() == "qns INFO value=7\n"
        assert messages == []
    finally:
        log.logger.removeHandler(handler)
        handler.close()


def test_exception_and_stack_info(messages):
    try:
        raise ValueError("original error")
    except ValueError:
        log.logger.exception("failed %s", "operation")
    record = messages[0].record
    assert record["message"] == "failed operation"
    assert record["exception"].type is ValueError
    assert "ValueError: original error" in str(messages[0])
    log.logger.info("with stack", stack_info=True)
    assert "Stack (most recent call last):" in messages[1].record["message"]


def test_custom_default_handler_formatter(messages):
    log.handle.setFormatter(logging.Formatter("%(levelname)s %(run_id)s %(message)s"))
    try:
        raise ValueError("formatted error")
    except ValueError:
        log.logger.error("failed %d", 3, exc_info=True, extra={"run_id": "run-1"})
    assert len(messages) == 1
    assert messages[0].record["message"].startswith("ERROR run-1 failed 3\nTraceback")
    assert str(messages[0]).count("ValueError: formatted error") == 1


def test_application_loguru_sink_remains_usable(messages):
    loguru_logger.info("application {value}", value=1)
    log.info("simulator %d", 2)
    assert [m.record["message"] for m in messages] == ["application 1", "simulator 2"]


@pytest.mark.parametrize("seconds,expected", [
    (0, "0.000s"), (0.001, "0.001s"),
    (1.999, "1.999s"), (65.123, "65.123s"),
    (3661.5, "3661.500s"), (86401.25, "86401.250s"),
])
def test_format_uses_simulation_time(messages, seconds, expected):
    simulator = Simulator(seconds, seconds + 1)
    log.install(simulator)
    stream = io.StringIO()
    sink = loguru_logger.add(stream, format=log.format_record, colorize=False)
    try:
        log.logger.info("simulated message")
    finally:
        loguru_logger.remove(sink)
    record = messages[0].record
    assert stream.getvalue().startswith(
        "[" + format(record["time"], "HH:mm:ss.SSS") + f"] sim={expected} | INFO"
    )
    assert " [test_log:" in stream.getvalue()
    assert stream.getvalue().endswith("] simulated message\n")
    assert record["extra"]["sim_time"] == expected


def test_simulation_timestamp_marks_missing_simulator(messages):
    log.info("before installation")
    assert messages[0].record["extra"]["sim_time"] == "—"
    assert "sim=" not in str(messages[0])
    assert "] | INFO " in str(messages[0])


def test_simulation_timestamp_advances_with_events(messages):
    simulator = Simulator(10, 100)
    log.install(simulator)
    log.logger.info("before event")
    simulator.add_event(func_to_event(
        simulator.time(sec=65.123), lambda: log.logger.info("during event")
    ))
    simulator.run()
    times = {m.record["message"]: m.record["extra"]["sim_time"] for m in messages}
    assert times["before event"] == "10.000s"
    assert times["during event"] == "65.123s"


def test_system_timestamps_are_preserved_for_both_loggers(messages):
    log.install(Simulator(65.123, 100))
    loguru_logger.info("application clock")
    log.logger.info("simulation clock")
    for message in messages:
        timestamp = message.record["time"]
        assert abs((datetime.now(timestamp.tzinfo) - timestamp).total_seconds()) < 5
        assert message.record["extra"]["sim_time"] == "65.123s"


def test_default_format_avoids_duplicate_time_but_preserves_legacy_message(messages):
    log.install(Simulator(65.123, 100))
    stream = io.StringIO()
    sink = loguru_logger.add(stream, format=log.format_record, colorize=False)
    try:
        log.info("value=%d", 7)
    finally:
        loguru_logger.remove(sink)
    assert "] sim=65.123s | INFO " in stream.getvalue()
    assert stream.getvalue().endswith("] value=7\n")
    assert messages[0].record["message"] == "[65.123] value=7"


def test_native_loguru_uses_same_format_without_simulator(messages):
    stream = io.StringIO()
    sink = loguru_logger.add(stream, format=log.format_record, colorize=False)
    try:
        loguru_logger.info("application {}", 1)
    finally:
        loguru_logger.remove(sink)
    assert "sim=" not in stream.getvalue()
    assert "] | INFO " in stream.getvalue()
    assert stream.getvalue().endswith("] application 1\n")


def _run_config_script(source):
    # Isolate configure(), which intentionally replaces application-wide sinks.
    return subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(source)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True,
    )


@pytest.mark.parametrize("output,terminal,file", [
    ("console", True, False), ("file", False, True), ("both", True, True),
])
def test_configure_destinations_and_debug_level(tmp_path, output, terminal, file):
    filename = tmp_path / "logs" / "experiment.log"
    result = _run_config_script(f"""
        from pathlib import Path
        import qns.utils.log as log
        from qns.simulator import Simulator

        log.install(Simulator(65.123, 100))
        log.configure({output!r}, Path({str(filename)!r}), level="DEBUG")
        log.debug("debug value=%d", 7)
        log.info("info value=%d", 8)
    """)
    assert result.stdout == ""
    assert filename.exists() is file
    destinations = []
    if terminal:
        destinations.append(result.stderr)
    else:
        assert result.stderr == ""
    if file:
        destinations.append(filename.read_text(encoding="utf-8"))
    for text in destinations:
        assert text.count("debug value=7") == 1
        assert text.count("info value=8") == 1
        assert "sim=65.123s" in text
        assert "[65.123]" not in text
        assert "\x1b[" not in text


def test_configure_replaces_outputs_and_overwrites_files(tmp_path):
    filename = tmp_path / "experiment.log"
    filename.write_text("old log content\n", encoding="utf-8")
    result = _run_config_script(f"""
        import logging
        import qns.utils.log as log

        log.configure("file", {str(filename)!r}, level="INFO")
        log.info("first file entry")
        log.configure("file", {str(filename)!r}, level=logging.DEBUG)
        log.debug("second file entry")
        log.configure("console", level="ERROR")
        log.info("suppressed info")
        log.error("only terminal entry")
    """)
    text = filename.read_text(encoding="utf-8")
    assert "old log content" not in text
    assert "first file entry" not in text
    assert text.count("second file entry") == 1
    assert "only terminal entry" not in text
    assert result.stderr.count("only terminal entry") == 1
    assert "file entry" not in result.stderr
    assert "suppressed info" not in result.stderr
    assert "sim=" not in result.stderr


def test_configure_invalid_arguments_preserve_existing_output():
    result = _run_config_script("""
        import qns.utils.log as log

        log.configure("console", level="INFO")
        for kwargs in [
            {"output": "invalid"}, {"output": "file"}, {"output": "both", "filename": ""},
            {"level": "invalid"}, {"level": -1}, {"level": None},
        ]:
            try:
                log.configure(**kwargs)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError("Invalid arguments should be rejected")
        log.info("still configured")
    """)
    assert result.stderr.count("still configured") == 1
