"""
Logging setup for the AEB project.
"""

import os
import sys
import logging
import time
from datetime import datetime

def setup_logger(name, log_file, level=logging.INFO):
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
    
    Returns:
        Configured logger
    """
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Set up logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Add handlers if they don't already exist
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def get_timestamp():
    """Get a timestamp string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

class Timer:
    """Simple timer class for logging execution time."""
    
    def __init__(self, name="Operation", logger=None):
        self.name = name
        self.logger = logger
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        
        if self.logger:
            if exc_type is not None:
                self.logger.error(f"{self.name} failed after {elapsed_time:.2f} seconds with error: {exc_val}")
            else:
                self.logger.info(f"Completed {self.name} in {elapsed_time:.2f} seconds")
        
        return False  # Don't suppress exceptions
    
    def elapsed(self):
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time