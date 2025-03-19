"""Result storage functionality."""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultStorage:
    """
    Handles storage and retrieval of experiment results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the result storage.
        
        Args:
            config: Result storage configuration
        """
        self.config = config
        self.format = config.get("format", "json")
        self.directory = config.get("directory", "./results")
        
        # Create directory if it doesn't exist
        os.makedirs(self.directory, exist_ok=True)
    
    def store(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store experiment results and analysis.
        
        Args:
            results: List of experiment results
            analysis: Analysis of results
            metadata: Optional metadata about the experiment
            
        Returns:
            Path to the saved file
        """
        # Create timestamp for filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create filename
        filename = metadata.get("name", "experiment").replace(" ", "_")
        filepath = os.path.join(self.directory, f"{filename}_{timestamp}.json")
        
        # Prepare data object
        data = {
            "metadata": metadata or {},
            "timestamp": timestamp,
            "results": results,
            "analysis": analysis
        }
        
        # Add timestamp to metadata
        data["metadata"]["timestamp"] = timestamp
        
        # Save to file based on format
        if self.format.lower() == "json":
            try:
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Results saved to {filepath}")
                return filepath
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                return ""
        else:
            # Other formats not implemented yet
            logger.warning(f"Format {self.format} not supported, using JSON")
            try:
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Results saved to {filepath}")
                return filepath
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                return ""
    
    def load(self, filepath: str) -> Dict[str, Any]:
        """
        Load results from a file.
        
        Args:
            filepath: Path to the results file
            
        Returns:
            Loaded results data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is invalid
        """
        if not os.path.exists(filepath):
            logger.error(f"Results file not found: {filepath}")
            raise FileNotFoundError(f"Results file not found: {filepath}")
            
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing results file: {e}")
            raise ValueError(f"Invalid results file: {e}")
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise ValueError(f"Error loading results: {e}")
    
    def list_results(self) -> List[str]:
        """
        List all result files in the configured directory.
        
        Returns:
            List of result file paths
        """
        if not os.path.exists(self.directory):
            return []
            
        # Find all JSON files (assuming JSON format)
        result_files = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".json"):
                result_files.append(os.path.join(self.directory, filename))
                
        # Sort by modification time (newest first)
        result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        return result_files
