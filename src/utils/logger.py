import logging
import sys
import os

def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a configured logger with the structured formatter:
    [TIMESTAMP] [MODULE_NAME] [LEVEL] - Message
    Outputs to both stdout and a persistent pipeline_debug.log file.
    """
    logger = logging.getLogger(module_name)

    # prevent adding handlers multiple times if the logger is requested again
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # streamHandler for console output
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # fileHandler for persistent log file in the project root
        # calculate the project root (assuming src/utils/logger.py is 2 levels deep)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        log_file_path = os.path.join(project_root, "pipeline_debug.log")

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
