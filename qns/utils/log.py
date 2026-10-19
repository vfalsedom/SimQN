#    SimQN: a discrete-event simulator for the quantum networks
#    Copyright (C) 2021-2022 Lutong Chen, Jian Li, Kaiping Xue
#    University of Science and Technology of China, USTC.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import sys

from loguru import logger as _loguru_logger


_FORMAT = (
    "<yellow>{time:HH:mm:ss.SSS}</yellow> | "
    "<yellow>sim={extra[sim_time]}</yellow> | "
    "<level>{level}</level> "
    "<cyan>[{name}:{line}]</cyan> "
    "<level>{extra[display_message]}</level>"
)


def _simulation_time():
    simulator = getattr(logger, "_simulator", None)
    if simulator is None:
        return "—"
    current_time = simulator.tc
    seconds = getattr(current_time, "sec", current_time)
    return f"{seconds:.3f}s"


def _patch_simulation_time(record):
    """Attach simulation time without replacing Loguru's system timestamp."""
    record["extra"]["sim_time"] = _simulation_time()


def format_record(record):
    """Format both SimQN and native Loguru records with the two clocks."""
    extra = record["extra"]
    extra.setdefault("sim_time", _simulation_time())
    message = record["message"]
    original = extra.get("logging_record")
    prefix = getattr(original, "_qns_simulator_prefix", "")
    if prefix and message.startswith(prefix):
        # Keep the legacy message for other handlers, but display time only once.
        message = message[len(prefix):]
    extra["display_message"] = message
    return _FORMAT + "\n{exception}"


class _LoguruHandler(logging.Handler):
    """Forward standard logging records without changing the public Logger API."""

    def emit(self, record):
        try:
            try:
                level = _loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Logging uses %-style arguments, whereas Loguru uses str.format().
            # Resolve the message here and pass no formatting arguments to Loguru.
            if self.formatter is None:
                message = record.getMessage()
                exception = record.exc_info
                if record.stack_info:
                    message += "\n" + record.stack_info
            else:
                message = self.format(record)
                exception = None  # Formatter already includes the traceback.

            frame, depth = sys._getframe(), 0
            while frame is not None and frame.f_code.co_filename in (logging.__file__, __file__):
                frame = frame.f_back
                depth += 1

            _loguru_logger.patch(_patch_simulation_time).bind(logging_record=record).opt(
                depth=depth, exception=exception
            ).log(level, message)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


logger = logging.getLogger("qns")
"""
The default ``logger`` used by SimQN, with a Loguru output backend.

This remains a standard logging.Logger for compatibility with setLevel(),
handlers, filters, %-style arguments, and other existing logging APIs.
"""

logger.setLevel(logging.INFO)
handle = _LoguruHandler()
logger.addHandler(handle)

# Replace only Loguru's built-in sink; retain application-configured sinks.
# If the application already removed it, leave its configuration in control.
try:
    _loguru_logger.remove(0)
except ValueError:
    pass
else:
    _loguru_logger.add(sys.stderr, format=format_record)


def install(s):
    """
    Install the logger to the simulator

    Args:
        s (Simulator): the simulator
    """
    logger._simulator = s


def debug(msg, *args):
    if hasattr(logger, "_simulator"):
        prefix = f"[{logger._simulator.tc}] "
        logger.debug(prefix + msg, *args, extra={"_qns_simulator_prefix": prefix})
    else:
        logger.debug(msg, *args)


def info(msg, *args):
    if hasattr(logger, "_simulator"):
        prefix = f"[{logger._simulator.tc}] "
        logger.info(prefix + msg, *args, extra={"_qns_simulator_prefix": prefix})
    else:
        logger.info(msg, *args)


def error(msg, *args):
    if hasattr(logger, "_simulator"):
        prefix = f"[{logger._simulator.tc}] "
        logger.error(prefix + msg, *args, extra={"_qns_simulator_prefix": prefix})
    else:
        logger.error(msg, *args)


def warn(msg, *args):
    if hasattr(logger, "_simulator"):
        prefix = f"[{logger._simulator.tc}] "
        logger.warning(prefix + msg, *args, extra={"_qns_simulator_prefix": prefix})
    else:
        logger.warning(msg, *args)


def critical(msg, *args):
    if hasattr(logger, "_simulator"):
        prefix = f"[{logger._simulator.tc}] "
        logger.critical(prefix + msg, *args, extra={"_qns_simulator_prefix": prefix})
    else:
        logger.critical(msg, *args)


def monitor(*args, sep: str = ",", with_time: bool = False):
    attrs = list(args)
    if with_time:
        attrs.insert(0, logger._simulator.tc)
    attrs_s = [str(a) for a in attrs]
    msg = sep.join(attrs_s)
    logger.info(msg)
