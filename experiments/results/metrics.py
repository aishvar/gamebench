"""Metrics calculation for experiments."""

import logging
from typing import Dict, Any, List, Optional, Union
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsTracker:
    """
    Calculates and tracks metrics for experiment evaluation.
    """
    
    def __init__(self, metric_names: List[str]):
        """
        Initialize the metrics tracker.
        
        Args:
            metric_names: List of metrics to calculate
        """
        self.metric_names = metric_names
        
        # Dictionary to store calculated metrics
        self.metrics = {}
        
        # Validate that all metrics are supported
        self._validate_metrics()
    
    def _validate_metrics(self):
        """Validate that all specified metrics are supported."""
        supported_metrics = {
            "win_rate", "avg_stack_change", "hands_played", 
            "decision_quality", "avg_decision_time"
        }
        
        for metric in self.metric_names:
            if metric not in supported_metrics:
                logger.warning(f"Unsupported metric: {metric}")
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate all specified metrics from results.
        
        Args:
            results: List of result dictionaries from experiment
            
        Returns:
            Dictionary of calculated metrics
        """
        # Reset metrics
        self.metrics = {}
        
        # Skip if no results
        if not results:
            logger.warning("No results to calculate metrics")
            return {}
            
        # Calculate each requested metric
        for metric in self.metric_names:
            try:
                if metric == "win_rate":
                    self.metrics["win_rate"] = self._calculate_win_rate(results)
                elif metric == "avg_stack_change":
                    self.metrics["avg_stack_change"] = self._calculate_avg_stack_change(results)
                elif metric == "hands_played":
                    self.metrics["hands_played"] = self._calculate_hands_played(results)
                elif metric == "decision_quality":
                    self.metrics["decision_quality"] = self._calculate_decision_quality(results)
                elif metric == "avg_decision_time":
                    self.metrics["avg_decision_time"] = self._calculate_avg_decision_time(results)
            except Exception as e:
                logger.error(f"Error calculating metric {metric}: {e}")
                self.metrics[metric] = {"error": str(e)}
                
        return self.metrics
    
    def analyze(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze results and calculate metrics.
        
        Args:
            results: List of result dictionaries from experiment
            
        Returns:
            Analysis with metrics and additional insights
        """
        # Calculate all metrics
        metrics = self.calculate_metrics(results)
        
        # Calculate additional analyses
        model_comparison = self._compare_models(results)
        summary_stats = self._calculate_summary_stats(results)
        
        # Combine into final analysis
        analysis = {
            "metrics": metrics,
            "model_comparison": model_comparison,
            "summary_stats": summary_stats
        }
        
        return analysis
    
    def _calculate_win_rate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate win rate for each model."""
        win_counts = {}
        total_games = len(results)
        
        for result in results:
            winner = result.get("winner")
            if winner:
                win_counts[winner] = win_counts.get(winner, 0) + 1
                
        # Calculate win rates as percentage
        win_rates = {}
        for model_id, wins in win_counts.items():
            win_rates[model_id] = (wins / total_games) * 100
            
        return win_rates
    
    def _calculate_avg_stack_change(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average stack change for each model."""
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
        """Calculate total hands played."""
        total_hands = 0
        
        for result in results:
            hands = result.get("hands_played", 0)
            total_hands += hands
            
        return total_hands
    
    def _calculate_decision_quality(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate decision quality metrics based on actions taken.
        This is a placeholder - actual implementation would depend on game-specific 
        metrics for evaluating decision quality.
        """
        # This would need game-specific logic to evaluate decision quality
        # For poker, might include metrics like:
        # - Percentage of good folds
        # - Percentage of good calls
        # - Percentage of good raises
        return {"note": "Decision quality metrics not implemented yet"}
    
    def _calculate_avg_decision_time(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average decision time for each model.
        This is a placeholder - actual implementation would require timing data
        in the results.
        """
        # This would need timing data in the results
        return {"note": "Average decision time metrics not implemented yet"}
    
    def _compare_models(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare models based on performance metrics."""
        # Extract all model IDs
        model_ids = set()
        for result in results:
            # Get models from player models dictionary
            player_models = result.get("player_models", {})
            for model_id in player_models.keys():
                model_ids.add(model_id)
        
        # Calculate direct matchup statistics
        matchups = {}
        for model_id in model_ids:
            matchups[model_id] = {}
            for opponent_id in model_ids:
                if model_id != opponent_id:
                    wins = 0
                    matches = 0
                    
                    # Count results where both these models played against each other
                    for result in results:
                        player_models = result.get("player_models", {})
                        players = list(player_models.keys())
                        
                        # Check if these two models played against each other
                        if len(players) == 2 and model_id in players and opponent_id in players:
                            matches += 1
                            if result.get("winner") == model_id:
                                wins += 1
                    
                    # Record matchup stats
                    if matches > 0:
                        matchups[model_id][opponent_id] = {
                            "matches": matches,
                            "wins": wins,
                            "win_rate": (wins / matches) * 100
                        }
        
        return {
            "model_ids": list(model_ids),
            "matchups": matchups
        }
    
    def _calculate_summary_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics for the experiment."""
        # Extract win margins
        win_margins = []
        for result in results:
            # Win margin is the difference between the winner's final stack and the loser's final stack
            players = result.get("players", {})
            if len(players) == 2:
                stacks = [player_data.get("final_stack", 0) for player_data in players.values()]
                if len(stacks) == 2:
                    margin = abs(stacks[0] - stacks[1])
                    win_margins.append(margin)
        
        # Calculate statistics
        summary = {}
        if win_margins:
            summary["win_margins"] = {
                "mean": statistics.mean(win_margins) if win_margins else 0,
                "median": statistics.median(win_margins) if win_margins else 0,
                "min": min(win_margins) if win_margins else 0,
                "max": max(win_margins) if win_margins else 0
            }
            
        return summary