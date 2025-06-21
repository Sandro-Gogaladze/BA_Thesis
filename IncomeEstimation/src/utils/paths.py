"""
Path utilities for the IncomeEstimation project.
"""

from pathlib import Path

def get_project_root():
    """
    Get the absolute path to the IncomeEstimation project root directory.
    
    Returns
    -------
    Path
        Path to the project root directory
    """
    return Path(__file__).resolve().parents[2]

def get_data_dir():
    """
    Get the absolute path to the data directory, which is at the same level as IncomeEstimation.
    
    Returns
    -------
    Path
        Path to the data directory
    """
    return get_project_root().parent / "data"

def get_raw_data_dir():
    """
    Get the absolute path to the raw data directory.
    
    Returns
    -------
    Path
        Path to the raw data directory
    """
    return get_data_dir() / "raw"

def get_processed_data_dir():
    """
    Get the absolute path to the processed data directory.
    
    Returns
    -------
    Path
        Path to the processed data directory
    """
    return get_data_dir() / "processed"

def resolve_data_path(data_path):
    """
    Resolve a data path which may be relative to the data directory.
    
    Parameters
    ----------
    data_path : str or Path
        Path to data, which may be relative to the data directory
        
    Returns
    -------
    Path
        Absolute path to the data file
    """
    data_path = Path(data_path)
    
    # If it's already an absolute path, return it
    if data_path.is_absolute():
        return data_path
    
    # If it starts with 'data/', assume it's relative to the project's parent
    if str(data_path).startswith('data/'):
        return get_project_root().parent / data_path
    
    # Otherwise, assume it's relative to the data directory
    return get_data_dir() / data_path
