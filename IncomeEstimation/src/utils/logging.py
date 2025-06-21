"""
Logging utilities for the IncomeEstimation project.

This module provides standardized logging functions for all components
of the IncomeEstimation project, ensuring consistent log formatting
and organization across different modules.
"""

import os
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(component_name, log_level=logging.INFO):
    """
    Set up a logger for a specific component of the application.
    All components log to a single file called income_estimation.log.
    
    Parameters
    ----------
    component_name : str
        Name of the component (e.g., 'preprocessing', 'training', 'inference')
        This will be included in the log messages.
    log_level : int, default=logging.INFO
        Logging level (e.g., logging.DEBUG, logging.INFO, etc.)
        
    Returns
    -------
    logging.Logger
        A configured logger instance
    """
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent.absolute()
    
    # Create logs directory if it doesn't exist
    logs_dir = project_root / 'logs'
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True)
    
    # Create logger
    logger = logging.getLogger(component_name)
    
    # Only add handlers if they haven't been added already
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # Use single log file for all components
        log_file = logs_dir / "income_estimation.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(component_name, log_level=logging.INFO):
    """
    Get a logger for a specific component.
    
    Parameters
    ----------
    component_name : str
        Name of the component (e.g., 'preprocessing', 'training', 'inference')
    log_level : int, default=logging.INFO
        Logging level (e.g., logging.DEBUG, logging.INFO, etc.)
        
    Returns
    -------
    logging.Logger
        A configured logger instance
    """
    return setup_logger(component_name, log_level)
