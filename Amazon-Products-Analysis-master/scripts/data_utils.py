"""
Utility functions for working with Amazon Q/A dataset
"""

import gzip
import json
import pandas as pd
from pathlib import Path
from typing import Iterator, Dict, Any, Optional


def parse_gzip_json(path: str) -> Iterator[Dict[str, Any]]:
    """
    Parse gzipped JSON file where each line is a JSON object.
    
    Args:
        path: Path to the .json.gz file
        
    Yields:
        Dictionary objects parsed from each line
    """
    with gzip.open(path, 'rt', encoding='utf-8') as g:
        for line in g:
            yield eval(line)  # Using eval as per the dataset documentation


def parse_strict_json(path: str) -> Iterator[Dict[str, Any]]:
    """
    Parse gzipped JSON file using strict JSON parsing (no eval).
    
    Args:
        path: Path to the .json.gz file
        
    Yields:
        Dictionary objects parsed from each line
    """
    with gzip.open(path, 'rt', encoding='utf-8') as g:
        for line in g:
            yield json.loads(line)


def load_dataframe(path: str, use_strict_json: bool = False) -> pd.DataFrame:
    """
    Load Amazon Q/A data into a pandas DataFrame.
    
    Args:
        path: Path to the .json.gz file
        use_strict_json: If True, use strict JSON parsing. If False, use eval (faster but less safe)
        
    Returns:
        pandas DataFrame containing the Q/A data
    """
    parser = parse_strict_json if use_strict_json else parse_gzip_json
    data = []
    
    for entry in parser(path):
        data.append(entry)
    
    return pd.DataFrame(data)


def convert_to_strict_json(input_path: str, output_path: str):
    """
    Convert the dataset from eval-format to strict JSON format.
    
    Args:
        input_path: Path to input .json.gz file
        output_path: Path to output .json file (will be uncompressed)
    """
    with gzip.open(input_path, 'rt', encoding='utf-8') as g_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in g_in:
            # Parse using eval, then write as strict JSON
            data = eval(line)
            json.dump(data, f_out)
            f_out.write('\n')


def get_category_files(data_dir: str = "data") -> list:
    """
    Get list of all available category files in the data directory.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of file paths
    """
    data_path = Path(data_dir)
    return sorted(data_path.glob("qa_*.json.gz"))


def load_category(category: str, data_dir: str = "data", use_strict_json: bool = False) -> pd.DataFrame:
    """
    Load a specific category's Q/A data.
    
    Args:
        category: Category name (e.g., "Electronics", "Beauty")
        data_dir: Path to data directory
        use_strict_json: If True, use strict JSON parsing
        
    Returns:
        pandas DataFrame containing the category's Q/A data
    """
    data_path = Path(data_dir)
    filename = f"qa_{category}.json.gz"
    filepath = data_path / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Category file not found: {filepath}")
    
    return load_dataframe(str(filepath), use_strict_json=use_strict_json)


# Example usage
if __name__ == "__main__":
    # Example: Load Electronics category
    data_dir = Path(__file__).parent / "data"
    
    # List all available files
    print("Available category files:")
    for file in get_category_files(str(data_dir)):
        print(f"  - {file.name}")
    
    # Example: Load a specific category
    print("\n" + "="*60)
    print("Example: Loading Electronics category")
    print("="*60)
    
    try:
        df = load_category("Electronics", str(data_dir))
        print(f"\nLoaded {len(df)} questions")
        print("\nFirst few entries:")
        print(df.head())
        print("\nDataFrame info:")
        print(df.info())
    except FileNotFoundError as e:
        print(f"Error: {e}")

