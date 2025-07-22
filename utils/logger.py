"""
Enhanced logging utilities
"""
import logging
import os
from config import LOG_LEVEL, LOG_FORMAT, DATA_DIR

def setup_logger(name: str) -> logging.Logger:
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    if not logger.handlers:
        os.makedirs(f"{DATA_DIR}/logs", exist_ok=True)
        
        file_handler = logging.FileHandler(f"{DATA_DIR}/logs/{name}.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, LOG_LEVEL))
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger
