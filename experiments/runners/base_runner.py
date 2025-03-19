"""Base experiment runner class."""

from abc import ABC, abstractmethod
import logging
import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentRunner(ABC):
    """
    Abstract base class for experiment runners.
    Defines the interface for experiment execution.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.start_time = None
        self.end_time = None
        self.results = []
        self.analysis = {}
        
        # Create output directory if it doesn't exist
        output_dir = self.config.get("output", {}).get("directory", "./results")
        os.makedirs(output_dir, exist_ok=True)
    
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Execute the experiment.
        
        Returns:
            Analysis of experiment results
        """
        pass
    
    @abstractmethod
    def _initialize_experiment(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Initialize game and models for the experiment.
        
        Returns:
            Tuple of (games, models) dictionaries
        """
        pass
    
    @abstractmethod
    def _run_iteration(self, 
                      iteration: int, 
                      games: Dict[str, Any], 
                      models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single iteration of the experiment.
        
        Args:
            iteration: Iteration number
            games: Dictionary of game instances
            models: Dictionary of model instances
            
        Returns:
            Results for this iteration
        """
        pass
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze experiment results.
        
        Args:
            results: List of results from iterations
            
        Returns:
            Analysis of results
        """
        if not results:
            return {"error": "No results to analyze"}
            
        try:
            # Extract metrics from config
            metrics = self.config.get("metrics", ["win_rate"])
            analysis = {
                "experiment": {
                    "name": self.config.get("name", "unknown"),
                    "total_iterations": len(results),
                    "duration_seconds": self.end_time - self.start_time if self.end_time else None
                },
                "metrics": {}
            }
            
            # Calculate metrics
            for metric in metrics:
                # Specific metric calculation based on type
                if metric == "win_rate":
                    analysis["metrics"]["win_rate"] = self._calculate_win_rate(results)
                elif metric == "avg_stack_change":
                    analysis["metrics"]["avg_stack_change"] = self._calculate_avg_stack_change(results)
                elif metric == "hands_played":
                    analysis["metrics"]["hands_played"] = self._calculate_hands_played(results)
                    
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing results: {e}")
            return {"error": f"Error analyzing results: {e}"}
    
    def _calculate_win_rate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate win rate for each model.
        
        Args:
            results: List of results from iterations
            
        Returns:
            Dictionary mapping model IDs to win rates
        """
        win_counts = {}
        total_games = len(results)
        
        for result in results:
            winner = result.get("winner")
            if winner:
                win_counts[winner] = win_counts.get(winner, 0) + 1
                
        # Calculate win rates
        win_rates = {}
        for model_id, wins in win_counts.items():
            win_rates[model_id] = wins / total_games
            
        return win_rates
    
    def _calculate_avg_stack_change(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average stack change for each model.
        
        Args:
            results: List of results from iterations
            
        Returns:
            Dictionary mapping model IDs to average stack changes
        """
        stack_changes = {}
        counts = {}
        
        for result in results:
            players = result.get("players", {})
            for player_id, player_data in players.items():
                if "net_winnings" in player_data:
                    if player_id not in stack_changes:
                        stack_changes[player_id] = 0
                        counts[player_id] = 0
                    stack_changes[player_id] += player_data["net_winnings"]
                    counts[player_id] += 1
                    
        # Calculate averages
        avg_changes = {}
        for player_id, total_change in stack_changes.items():
            if counts[player_id] > 0:
                avg_changes[player_id] = total_change / counts[player_id]
                
        return avg_changes
    
    def _calculate_hands_played(self, results: List[Dict[str, Any]]) -> int:
        """
        Calculate the total number of hands played.
        
        Args:
            results: List of results from iterations
            
        Returns:
            Total number of hands played
        """
        total_hands = 0
        
        for result in results:
            hands = result.get("hands_played", 0)
            total_hands += hands
            
        return total_hands
    
    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """
        Save results and analysis to file.
        
        Args:
            results: List of results from iterations
            analysis: Analysis of results
            
        Returns:
            Path to the saved file
        """
        # Create timestamp for filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Determine output format and directory
        output_format = self.config.get("output", {}).get("format", "json")
        output_dir = self.config.get("output", {}).get("directory", "./results")
        experiment_name = self.config.get("name", "experiment").replace(" ", "_")
        
        # Create results object with experiment metadata
        output_data = {
            "experiment": {
                "name": self.config.get("name", "unknown"),
                "description": self.config.get("description", ""),
                "timestamp": timestamp,
                "iterations": len(results),
                "duration_seconds": self.end_time - self.start_time if self.end_time else None
            },
            "results": results,
            "analysis": analysis
        }
        
        # Save to file
        if output_format.lower() == "json":
            filepath = os.path.join(output_dir, f"{experiment_name}_{timestamp}.json")
            try:
                with open(filepath, "w") as f:
                    json.dump(output_data, f, indent=2)
                logger.info(f"Results saved to {filepath}")
                return filepath
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                return ""
        else:
            # CSV format not implemented yet
            logger.warning(f"Output format '{output_format}' not supported yet, using JSON")
            # Default to JSON
            filepath = os.path.join(output_dir, f"{experiment_name}_{timestamp}.json")
            try:
                with open(filepath, "w") as f:
                    json.dump(output_data, f, indent=2)
                logger.info(f"Results saved to {filepath}")
                return filepath
            except Exception as e:
                logger.error(f"Error saving results: {e}")
                return ""
