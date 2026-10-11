import logging

from lightning.pytorch.utilities import rank_zero_only


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Create a logger that emits only from rank zero."""
    logger = logging.getLogger(name)

    # Log once per GPU group.
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
