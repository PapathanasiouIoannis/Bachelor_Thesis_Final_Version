import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a cleanly formatted, timestamped logger for the given module name.
    Outputs to standard output with level INFO by default.
    """
    logger = logging.getLogger(name)
    
    # prevent adding multiple handlers if the logger is requested multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    return logger
