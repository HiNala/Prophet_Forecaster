"""
Data management module for handling file organization and versioning.
Provides utilities for saving and loading data, models, and results.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import pandas as pd
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

class DataManager:
    def __init__(self, base_dir: str = "."):
        """Initialize data manager with base directory structure."""
        self.base_dir = Path(base_dir)
        
        # Define directory structure
        self.dirs = {
            'data': {
                'raw': self.base_dir / 'data' / 'raw',
                'interim': self.base_dir / 'data' / 'interim',
                'features': self.base_dir / 'data' / 'features',
                'processed': self.base_dir / 'data' / 'processed',
                'visualizations': self.base_dir / 'data' / 'visualizations'
            },
            'models': {
                'trained': self.base_dir / 'models' / 'trained',
                'metadata': self.base_dir / 'models' / 'metadata',
                'evaluation': self.base_dir / 'models' / 'evaluation'
            },
            'forecasts': {
                'predictions': self.base_dir / 'forecasts' / 'predictions',
                'visualizations': self.base_dir / 'forecasts' / 'visualizations',
                'evaluation': self.base_dir / 'forecasts' / 'evaluation'
            }
        }
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Initialize version tracking
        self.version_file = self.base_dir / 'version_tracking.json'
        self.versions = self._load_versions()

    def _create_directories(self):
        """Create the directory structure if it doesn't exist."""
        for category in self.dirs.values():
            for dir_path in category.values():
                dir_path.mkdir(parents=True, exist_ok=True)

    def _load_versions(self) -> Dict[str, Any]:
        """Load version tracking information."""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {'data': {}, 'models': {}, 'forecasts': {}}

    def _save_versions(self):
        """Save version tracking information."""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=4)

    def _generate_version_id(self, category: str, prefix: str) -> str:
        """Generate a unique version ID for a file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version = self.versions.get(category, {}).get(prefix, 0) + 1
        self.versions.setdefault(category, {})[prefix] = version
        self._save_versions()
        return f"{prefix}_{timestamp}_v{version:03d}"

    def save_data(
        self,
        data: Union[pd.DataFrame, Dict, Any],
        category: str,
        subcategory: str,
        prefix: str,
        extension: str = 'csv',
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save data with versioning.
        
        Args:
            data: Data to save (DataFrame, dict, or other serializable object)
            category: Main category ('data', 'models', 'forecasts')
            subcategory: Subcategory (e.g., 'raw', 'interim', 'features')
            prefix: File prefix for versioning
            extension: File extension (default: 'csv')
            metadata: Optional metadata to save alongside the data
            
        Returns:
            str: Path to saved file
        """
        try:
            # Generate version ID
            version_id = self._generate_version_id(category, prefix)
            
            # Construct file path
            file_path = self.dirs[category][subcategory] / f"{version_id}.{extension}"
            
            # Save data based on type
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            elif isinstance(data, dict) or isinstance(data, list):
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
            else:
                with open(file_path, 'wb') as f:
                    import pickle
                    pickle.dump(data, f)
            
            # Save metadata if provided
            if metadata:
                metadata_path = self.dirs[category][subcategory] / f"{version_id}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            
            logger.info(f"Saved {category}/{subcategory}/{version_id}.{extension}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def load_data(
        self,
        category: str,
        subcategory: str,
        version_id: Optional[str] = None,
        prefix: Optional[str] = None,
        extension: str = 'csv'
    ) -> Any:
        """
        Load data by version ID or latest version with given prefix.
        
        Args:
            category: Main category ('data', 'models', 'forecasts')
            subcategory: Subcategory (e.g., 'raw', 'interim', 'features')
            version_id: Specific version ID to load
            prefix: File prefix to find latest version
            extension: File extension (default: 'csv')
            
        Returns:
            Loaded data
        """
        try:
            dir_path = self.dirs[category][subcategory]
            
            # Find the file to load
            if version_id:
                file_path = dir_path / f"{version_id}.{extension}"
            elif prefix:
                pattern = f"{prefix}_*.{extension}"
                files = list(dir_path.glob(pattern))
                if not files:
                    raise FileNotFoundError(f"No files found matching pattern {pattern}")
                file_path = max(files, key=lambda x: x.stat().st_mtime)
            else:
                raise ValueError("Either version_id or prefix must be provided")
            
            # Load data based on extension
            if extension == 'csv':
                return pd.read_csv(file_path)
            elif extension == 'json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                with open(file_path, 'rb') as f:
                    import pickle
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def get_latest_version(
        self,
        category: str,
        subcategory: str,
        prefix: str,
        extension: str = 'csv'
    ) -> Optional[str]:
        """Get the latest version ID for a given prefix."""
        try:
            pattern = f"{prefix}_*.{extension}"
            files = list(self.dirs[category][subcategory].glob(pattern))
            if not files:
                return None
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            return latest_file.stem
            
        except Exception as e:
            logger.error(f"Error getting latest version: {str(e)}")
            raise

    def list_versions(
        self,
        category: str,
        subcategory: str,
        prefix: Optional[str] = None,
        extension: str = 'csv'
    ) -> List[str]:
        """List all versions for a given category/subcategory/prefix."""
        try:
            pattern = f"{prefix}_*.{extension}" if prefix else f"*.{extension}"
            files = list(self.dirs[category][subcategory].glob(pattern))
            return [f.stem for f in sorted(files, key=lambda x: x.stat().st_mtime)]
            
        except Exception as e:
            logger.error(f"Error listing versions: {str(e)}")
            raise

    def cleanup_old_versions(
        self,
        category: str,
        subcategory: str,
        prefix: str,
        keep_last_n: int = 5,
        extension: str = 'csv'
    ):
        """Clean up old versions, keeping only the last N versions."""
        try:
            pattern = f"{prefix}_*.{extension}"
            files = list(self.dirs[category][subcategory].glob(pattern))
            files.sort(key=lambda x: x.stat().st_mtime)
            
            # Keep the latest n files
            files_to_delete = files[:-keep_last_n] if len(files) > keep_last_n else []
            
            for file in files_to_delete:
                # Remove data file
                file.unlink()
                
                # Remove metadata if exists
                metadata_file = file.parent / f"{file.stem}_metadata.json"
                if metadata_file.exists():
                    metadata_file.unlink()
                    
            if files_to_delete:
                logger.info(f"Cleaned up {len(files_to_delete)} old versions of {prefix}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {str(e)}")
            raise

def initialize() -> DataManager:
    """Initialize the data manager."""
    return DataManager() 