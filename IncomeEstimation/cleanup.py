#!/usr/bin/env python
"""
Cleanup script to delete all generated outputs from training, prediction, and evaluation.
This includes figures, metrics, models, predictions, and any other generated files.
"""
import os
import shutil
import argparse
from pathlib import Path
from IncomeEstimation.src.utils.paths import get_project_root

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Clean up generated model files")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="all",
        choices=["xgboost", "segment_aware", "huber_threshold", "quantile", "all"],
        help="Model type to clean up, or 'all' for all models"
    )
    
    parser.add_argument(
        "--confirm", 
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    return parser.parse_args()

def cleanup_model_outputs(model_type, confirm=False):
    """
    Clean up all generated outputs for a specific model
    
    Parameters:
    -----------
    model_type : str
        Model type to clean up ('xgboost', 'segment_aware', 'huber_threshold', or 'quantile')
    confirm : bool
        Whether to skip confirmation prompt
    """
    project_root = get_project_root()
    
    if model_type == "xgboost":
        results_dir = project_root / "baseline" / "xgboost" / "results"
        model_name = "XGBoost"
    elif model_type == "segment_aware":
        results_dir = project_root / "prehoc" / "segment_aware" / "results"
        model_name = "Segment-Aware"
    elif model_type == "huber_threshold":
        results_dir = project_root / "prehoc" / "huber_threshold" / "results"
        model_name = "Huber Threshold"
    elif model_type == "quantile":
        results_dir = project_root / "posthoc" / "quantile" / "results"
        model_name = "Quantile"
    else:
        print(f"Unknown model type: {model_type}")
        return
    
    if not results_dir.exists():
        print(f"No results directory found for {model_name} model. Nothing to clean up.")
        return
    
    # List subdirectories to clean up
    subdirs = ["figures", "metrics", "models", "predictions"]
    dirs_to_clean = []
    
    for subdir in subdirs:
        subdir_path = results_dir / subdir
        if subdir_path.exists():
            dirs_to_clean.append(subdir_path)
    
    if not dirs_to_clean:
        print(f"No output files found for {model_name} model. Nothing to clean up.")
        return
    
    # Display what will be deleted
    print(f"\nThe following directories for {model_name} model will be cleaned:")
    for directory in dirs_to_clean:
        print(f" - {directory}")
    print(f"This will delete all files in these directories.")
    
    # Confirm before proceeding
    if not confirm:
        response = input("\nDo you want to proceed? [y/N]: ").lower()
        if response != 'y':
            print("Operation cancelled.")
            return
    
    # Clean up each subdirectory
    for directory in dirs_to_clean:
        try:
            # Delete files in directory but keep the directory structure
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    print(f"Deleted: {file_path}")
            print(f"Cleaned directory: {directory}")
        except Exception as e:
            print(f"Error cleaning {directory}: {str(e)}")

def main():
    """Main cleanup function"""
    args = parse_args()
    
    print("XGBoost Model Cleanup Utility")
    print("===========================")
    
    if args.model == "all":
        models = ["xgboost", "segment_aware", "huber_threshold", "quantile"]
        
        if not args.confirm:
            print("\nWarning: This will clean up ALL generated outputs for ALL models.")
            response = input("Do you want to proceed? [y/N]: ").lower()
            if response != 'y':
                print("Operation cancelled.")
                return
        
        for model in models:
            cleanup_model_outputs(model, confirm=True)
    else:
        cleanup_model_outputs(args.model, confirm=args.confirm)
    
    print("\nCleanup completed.")

if __name__ == "__main__":
    main()
