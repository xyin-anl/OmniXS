from .utils.logging import setup_logger

setup_logger(
    "omnixas",
    log_level="INFO",
    capture_prints=True,
    file_logging=False,
)

__all__ = ["setup_logger"]
