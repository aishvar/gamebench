"""Sequential experiment runner for turn-based games."""

import logging
import time
import os
import sys
from typing import Dict, Any, List, Tuple, Optional, Union
import random

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from game_engines.base_game import BaseGame
from game_engines.heads_up_poker import HeadsUpPoker, run_non_interactive_game
import game_engines.heads_up_poker
game_engines.heads_up_poker.logger = logger
from model_orchestrator.llm_client import LLMClient, parse_response_text
from model_orchestrator.game_adapter import GameAdapter

from experiments.runners.base_runner import ExperimentRunner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SequentialRunner(ExperimentRunner):
    """
    Sequential experiment runner for turn-based games.
    Runs experiments one at a time in a single thread.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sequential experiment runner.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config)
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the experiment.
        
        Returns:
            Analysis of experiment results
        """
        logger.info(f"Starting experiment: {self.config.get('name', 'unknown')}")
        self.start_time = time.time()
        
        try:
            # Initialize games and models
            games, models = self._initialize_experiment()
            
            # Run iterations
            iterations = self.config.get("iterations", 1)
            self.results = []
            
            for i in range(iterations):
                logger.info(f"Running iteration {i+1}/{iterations}")
                try:
                    # Run a single iteration
                    result = self._run_iteration(i, games, models)
                    self.results.append(result)
                except Exception as e:
                    logger.error(f"Error in iteration {i+1}: {e}")
                    # Continue with next iteration
            
            # Mark experiment end time
            self.end_time = time.time()
            
            # Analyze results
            self.analysis = self.analyze_results(self.results)
            
            # Save results
            self.save_results(self.results, self.analysis)
            
            logger.info(f"Experiment completed in {self.end_time - self.start_time:.2f} seconds")
            return self.analysis
            
        except Exception as e:
            self.end_time = time.time()
            logger.error(f"Experiment failed: {e}")
            return {"error": f"Experiment failed: {e}"}
    
    def _initialize_experiment(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Initialize game and models for the experiment.
        
        Returns:
            Tuple of (games, models) dictionaries
        """
        # Initialize games based on configuration
        games = {}
        game_config = self.config.get("game", {})
        game_type = game_config.get("type", "poker")
        
        if game_type == "poker":
            games["poker"] = self._init_poker_game(game_config)
        else:
            raise ValueError(f"Unsupported game type: {game_type}")
            
        # Initialize models based on configuration
        models = {}
        model_configs = self.config.get("models", [])
        
        if not model_configs:
            raise ValueError("No models specified in configuration")
            
        for i, model_config in enumerate(model_configs):
            model_id = f"model_{i+1}"
            models[model_id] = self._init_model(model_config, model_id)
            
        return games, models
    
    def _init_poker_game(self, config: Dict[str, Any]) -> HeadsUpPoker:
        """Initialize a poker game with configuration."""
        starting_stack = config.get("starting_stack", 1000)
        small_blind = config.get("small_blind", 10)
        big_blind = config.get("big_blind", 20)
        random_seed = config.get("random_seed")
        
        return HeadsUpPoker(
            random_seed=random_seed,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind
        )
    
    def _init_model(self, config: Dict[str, Any], model_id: str) -> Dict[str, Any]:
        """Initialize a model with configuration."""
        provider = config.get("provider", "openai")
        model_name = config.get("name", "gpt-4o")
        max_tokens = config.get("max_tokens", 1000)
        temperature = config.get("temperature", 0.7)
        max_retries = config.get("max_retries", 3)
        timeout = config.get("timeout", 60)
        
        llm_client = LLMClient(
            provider=provider,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout
        )
        
        return {
            "id": model_id,
            "config": config,
            "client": llm_client
        }
    
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
        # Reset game for this iteration
        game_type = self.config.get("game", {}).get("type", "poker")
        game = games[game_type]
        game.reset()
        
        # Create game adapter
        adapter = GameAdapter(game, game_type=game_type)
        
        # For poker, we'll run a non-interactive game with LLM agents
        if game_type == "poker":
            # We need at least 2 models for heads-up poker
            if len(models) < 2:
                # If only one model, play against itself
                model_ids = [list(models.keys())[0], list(models.keys())[0]]
            elif len(models) == 2:
                # Use both models
                model_ids = list(models.keys())
            else:
                # More than 2 models, randomly select 2
                model_ids = random.sample(list(models.keys()), 2)
            
            # Create agent functions for the selected models
            agents = []
            for model_id in model_ids:
                model = models[model_id]
                agent_fn = self._create_poker_agent(model, adapter, model_id)
                agents.append(agent_fn)
            
            # Configure game with player names matching model IDs
            game.player1_name = model_ids[0]
            game.player2_name = model_ids[1]
            game.reset()  # Reset again with new player names
            
            # Set game to non-interactive mode
            game.set_interactive_mode(False)
            
            # Run the game
            num_hands = self.config.get("num_hands", 10)
            result = run_non_interactive_game(
                player1_agent=agents[0],
                player2_agent=agents[1],
                num_hands=num_hands,
                random_seed=game.random_seed
            )
            
            # Add iteration metadata
            result["iteration"] = iteration + 1
            result["player_models"] = {
                model_ids[0]: models[model_ids[0]]["config"].get("name", "unknown"),
                model_ids[1]: models[model_ids[1]]["config"].get("name", "unknown")
            }
            result["hands_played"] = num_hands
            
            return result
        else:
            raise ValueError(f"Unsupported game type: {game_type}")
    
    def _create_poker_agent(self, model: Dict[str, Any], adapter: GameAdapter, model_id: str):
        """
        Create a poker agent function that can be passed to run_non_interactive_game.
        
        Args:
            model: Model instance dictionary
            adapter: GameAdapter instance
            model_id: ID of the model (used as player ID)
            
        Returns:
            Agent function that takes state and valid_actions and returns an action
        """
        import re
        # Ensure the logger is accessible here:
        logger = logging.getLogger(__name__)
        llm_client = model["client"]
        system_prompt = adapter.prepare_system_prompt()
    
        def agent_fn(state, valid_actions):
            # Skip LLM call if no valid actions are available
            if not valid_actions:
                logging.getLogger(__name__).warning(f"No valid actions available for {model_id}")
                return None
            
            # Create a new adapter for this specific prompt
            # This is to avoid any potential state conflicts
            from game_engines.heads_up_poker import HeadsUpPoker
            
            # Create a new game instance for this prompt
            new_game = HeadsUpPoker(
                random_seed=adapter.game.random_seed,
                player1_name=adapter.game.player1_name,
                player2_name=adapter.game.player2_name,
                starting_stack=adapter.game.starting_stack,
                small_blind=adapter.game.small_blind,
                big_blind=adapter.game.big_blind
            )
            
            # Update the new game's state based on the state parameter
            new_game.stage = state.get('stage', 'pre-flop')
            new_game.community_cards = state.get('community_cards', [])
            new_game.pot = state.get('pot', 0)
            new_game.current_bet = state.get('current_bet', 0)
            new_game.hand_number = state.get('hand_number', 1)
            
            # Find the active player index and dealer index
            new_game.dealer_index = 0 if state.get('dealer') == new_game.player1_name else 1
            if state.get('active_player'):
                new_game.active_player_index = 0 if state.get('active_player') == new_game.player1_name else 1
            
            # Update player information
            player_data = state.get('players', {})
            for i, player in enumerate(new_game.players_obj):
                player_info = player_data.get(player.name, {})
                player.stack = player_info.get('stack', new_game.starting_stack)
                player.current_bet = player_info.get('current_bet', 0)
                player.folded = player_info.get('folded', False)
                player.all_in = player_info.get('all_in', False)
                
                # Ensure hole cards are correctly set
                if 'hole_cards' in player_info:
                    player.hole_cards = player_info['hole_cards']
            
            # Create a new adapter with the updated game state
            temp_adapter = GameAdapter(new_game, game_type=adapter.game_type)
            
            # Format valid actions for inclusion in the prompt
            valid_actions_str = "Valid actions:\n"
            for action in valid_actions:
                action_type = action.get("action_type", "").capitalize()
                if action_type == "Fold":
                    valid_actions_str += "- Fold your hand\n"
                elif action_type == "Check":
                    valid_actions_str += "- Check (pass)\n"
                elif action_type == "Call":
                    amount = action.get("amount", 0)
                    valid_actions_str += f"- Call {amount} chips\n"
                elif action_type == "Raise":
                    min_amount = action.get("min_amount", 0)
                    max_amount = action.get("max_amount", "all-in")
                    valid_actions_str += f"- Raise between {min_amount} and {max_amount} chips\n"
            
            # Use the updated adapter to create the prompt
            prompt = temp_adapter.prepare_prompt(model_id)
            
            # Replace the valid actions section with our formatted valid actions
            # Use regex to replace the entire section, not just the heading
            prompt = re.sub(r"Valid actions you can take now:.*", valid_actions_str, prompt, flags=re.DOTALL)
            
            # Call LLM
            llm_response = llm_client.call_llm(
                developer_message="You are playing poker. Please select a valid action from the list of valid actions based on the current game state.",
                user_message=prompt,
                system_message=system_prompt
            )
            
            # Parse response text
            response_text = parse_response_text(llm_response)
            
            if not response_text:
                # If we couldn't parse a response, use fallback
                logging.getLogger(__name__).warning(f"Failed to parse response from {model_id}")
                return adapter.response_parser.get_fallback_action(valid_actions)
            
            # Parse and validate action
            try:
                action = adapter.parse_response(response_text, valid_actions)
                return action
            except Exception as e:
                logging.getLogger(__name__).error(f"Error parsing agent action for {model_id}: {e}")
                return adapter.response_parser.get_fallback_action(valid_actions)
                
        return agent_fn