"""
Logging Utilities

This module provides utilities for logging and timing operations.
"""

import os
import sys
import time
import logging
import functools
from typing import Optional, Callable, Any, Dict
from datetime import datetime

# Configure the logger
def setup_logger(
    log_file: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        log_file: Path to the log file. If None, no file handler is created.
        console_level: Logging level for console output
        file_level: Logging level for file output
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("concept_graph")
    
    # Clear existing handlers to avoid duplicate logging
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Set logger level to the minimum level to ensure all messages are processed
    logger.setLevel(min(console_level, file_level))
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized with console level: {logging.getLevelName(console_level)}")
    if log_file:
        logger.info(f"File logging enabled: {log_file} (level: {logging.getLevelName(file_level)})")
    
    return logger

# Decorator for timing functions
def timeit(func: Callable) -> Callable:
    """
    Decorator that logs the execution time of a function.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("concept_graph")
        start_time = time.time()
        
        logger.debug(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.debug(f"Completed {func.__name__} in {execution_time:.2f} seconds")
        
        # Add timing info to result if it's a dictionary
        if isinstance(result, dict):
            result['execution_time'] = execution_time
        
        return result
    
    return wrapper

# Class for maintaining an experiment log
class ExperimentLogger:
    """
    Class for maintaining an experiment log with timestamps and metrics.
    """
    
    def __init__(self, log_file: str):
        """
        Initialize the experiment logger.
        
        Args:
            log_file: Path to the log file
        """
        self.log_file = log_file
        self.start_time = datetime.now()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Initialize the log file with header
        with open(log_file, 'w') as f:
            f.write(f"Experiment Log - Started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n\n")
    
    def log_step(self, step_name: str, details: Optional[str] = None):
        """
        Log an experiment step with timestamp.
        
        Args:
            step_name: Name of the step
            details: Optional details about the step
        """
        timestamp = datetime.now()
        elapsed = timestamp - self.start_time
        
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] ")
            f.write(f"({elapsed.total_seconds():.2f}s) ")
            f.write(f"{step_name}")
            
            if details:
                f.write(f": {details}")
            
            f.write("\n")
    
    def log_metrics(self, metrics: Dict[str, Any], prefix: Optional[str] = None):
        """
        Log metrics to the experiment log.
        
        Args:
            metrics: Dictionary of metrics to log
            prefix: Optional prefix for the metrics
        """
        timestamp = datetime.now()
        elapsed = timestamp - self.start_time
        
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] ")
            f.write(f"({elapsed.total_seconds():.2f}s) ")
            
            if prefix:
                f.write(f"{prefix} - ")
            
            f.write("Metrics:\n")
            
            for key, value in metrics.items():
                f.write(f"  - {key}: {value}\n")
            
            f.write("\n")
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None):
        """
        Log an error to the experiment log.
        
        Args:
            error_message: Error message
            exception: Optional exception object
        """
        timestamp = datetime.now()
        elapsed = timestamp - self.start_time
        
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] ")
            f.write(f"({elapsed.total_seconds():.2f}s) ")
            f.write(f"ERROR: {error_message}")
            
            if exception:
                f.write(f" - {type(exception).__name__}: {str(exception)}")
            
            f.write("\n")
    
    def log_summary(self, summary_text: str):
        """
        Log a summary to the experiment log.
        
        Args:
            summary_text: Summary text
        """
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        with open(self.log_file, 'a') as f:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"Experiment Summary - Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {total_duration.total_seconds():.2f} seconds\n")
            f.write("\n")
            f.write(summary_text)
            f.write("\n" + "-" * 80 + "\n")