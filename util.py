import sys
import io
import chromadb
import kagglehub
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import os
from groq import Groq
from dotenv import load_dotenv


# ============================================================================
# Metadata Formatter
# ============================================================================


class SimpleMetadataFormatter:
    """
    Simple metadata formatter
    Formats metadata dict into readable text for context
    """

    def __init__(self, exclude_fields: Optional[List[str]] = None):
        """
        Args:
            exclude_fields: Fields to exclude from display
        """
        self.exclude_fields = exclude_fields or [
            'chunk_index', 'total_chunks', 'source_row']

    def format(self, metadata: Dict[str, Any]) -> str:
        """
        Format metadata into readable text

        Example:
            Input:  {'Title': 'Iron Man', 'Year': 2008, 'chunk_index': 0}
            Output: "Title: Iron Man\nYear: 2008"
        """
        lines = []

        for key, value in metadata.items():
            # Skip excluded fields and empty values
            if key in self.exclude_fields:
                continue
            if value is None or value == '' or value == 'N/A':
                continue

            # Format field name (replace underscore, capitalize)
            display_name = key.replace('_', ' ').title()
            lines.append(f"{display_name}: {value}")

        return '\n'.join(lines)

    def __call__(self, metadata: Dict[str, Any]) -> str:
        """Allow calling as function"""
        return self.format(metadata)


def download_data(source: str) -> str:
    """
    Download dataset from Kaggle.

    Args:
        source: Kaggle dataset identifier (e.g., "hgultekin/bbcnewsarchive")

    Returns:
        Path to downloaded dataset files
    """
    try:
        path = kagglehub.dataset_download(source)
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


def read_data(
        datapath: str = '_.csv',
        sep: str = '\t',
        debug: bool = False) -> pd.DataFrame:
    """
    Read CSV data into pandas DataFrame.

    Args:
        datapath: Path to CSV file
        sep: Delimiter character (default: tab)
        debug: Whether to print debug information

    Returns:
        pandas DataFrame containing the data
    """
    try:
        df = pd.read_csv(datapath, sep=sep)

        if debug:
            print(f"Successfully loaded {len(df)} articles")
            print(f"Columns: {df.columns.tolist()}")
            print(f"\nFirst article:\n{df.iloc[0]}")
            print(f"\nFirst title: {df.iloc[0]['content'][:100]}...")
            print(f"Content length: {len(df.iloc[0]['content'])} characters")

        return df
    except Exception as e:
        print(f"Error reading data: {e}")
        raise
