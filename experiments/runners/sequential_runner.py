# experiments/runners/sequential_runner.py

import logging
import time
import os
import sys
from typing import Dict, Any, List, Tuple, Optional, Union
import random
import copy
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)
    logger.debug(f"Added {project_root} to sys.path")

from game_engines.base_game import BaseGame
from game_engines.heads_up_poker import HeadsUpPoker, run_non_interactive_game # Import runner function
from model_orchestrator.utils import init_game_log, log_llm_call, log_event_to_game_file, parse_response_text
from model_orchestrator.prompt_templates import render_template # Ensure this import is present

from model_orchestrator.llm_client import LLMClient
from model_orchestrator.game_adapter import GameAdapter
from experiments.runners.base_runner import ExperimentRunner
from experiments.results import ResultStorage, MetricsTracker

class SequentialRunner(ExperimentRunner):
    """
    Sequential experiment runner for turn-based games.
    Runs experiments one at a time in a single thread.
    Each iteration runs a full game consisting of multiple independent hands.
    """

    def __init__(self, config: Dict[str, Any]):
        """ Initialize the sequential experiment runner. """
        super().__init__(config)
        self.metrics_tracker = MetricsTracker(config.get("metrics", []))
        self.result_storage = ResultStorage(config.get("output", {}))

        # Centralized log directory from config
        output_config = config.get("output", {})
        log_dir_base = output_config.get("directory", "./results")
        # Specific sub-directory for detailed game logs
        self.game_log_dir = os.path.join(log_dir_base, "game_logs")
        os.makedirs(self.game_log_dir, exist_ok=True)
        # Note: init_game_log is now called per iteration to create separate files
        logger.info(f"Game logs will be saved in: {self.game_log_dir}")

    # --- Implementation of Abstract Methods ---

    def _initialize_experiment(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Initialize models for the experiment.
        Game instances are created per iteration.

        Returns:
            Tuple of (games, models) dictionaries. Games dict is empty here.
        """
        logger.info("Initializing models for the experiment.")
        models = self._initialize_models()
        logger.info(f"Initialized {len(models)} models.")
        # No persistent game instances needed across iterations for this runner
        games = {}
        return games, models

    def _run_iteration(self, iteration: int, games: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single iteration (a full set of independent hands) of the experiment.

        Args:
            iteration: Iteration number (for logging and seeding).
            games: Dictionary of game instances (unused in this runner).
            models: Dictionary of initialized model data (client, config).

        Returns:
            Results dictionary for this iteration (aggregate game outcome).
        """
        game_config = self.config.get("game", {})
        game_type = game_config.get("type", "poker")

        if game_type != "poker":
            raise ValueError(f"Unsupported game type in SequentialRunner: {game_type}")

        # --- Initialize Game Log for this Iteration ---
        # Create a unique log file for each iteration
        iteration_log_path = init_game_log(self.game_log_dir)
        if not iteration_log_path:
             logger.error(f"Iteration {iteration}: Failed to initialize game log file. Continuing without detailed log.")
        log_event_to_game_file(f"\n===== EXPERIMENT: {self.config.get('name', 'unknown')} | ITERATION {iteration} =====\n")


        # --- Initialize Game Instance for this Iteration ---
        # Use a combination of base seed and iteration number for reproducibility
        base_seed = game_config.get("random_seed")
        iter_seed = hash((base_seed, iteration)) if base_seed is not None else None
        # Make seed generation robust
        if iter_seed is None:
            iter_seed = random.randint(0, 2**32 - 1) # Generate a random seed if base is None
            logger.warning(f"Iteration {iteration}: No base random_seed provided, using generated seed: {iter_seed}")


        player_ids = list(models.keys())
        if len(player_ids) != 2:
            raise ValueError(f"SequentialRunner expects exactly 2 models, found {len(player_ids)}")
        player1_id = player_ids[0]
        player2_id = player_ids[1]

        # Create the game instance specifically for this iteration
        game = HeadsUpPoker(
            game_id=f"iter_{iteration}_{int(time.time())}", # Unique game ID per iteration
            player1_name=player1_id,
            player2_name=player2_id,
            starting_stack=game_config.get("starting_stack", 1000),
            small_blind=game_config.get("small_blind", 10),
            big_blind=game_config.get("big_blind", 20),
            random_seed=iter_seed # Use iteration-specific seed
        )
        game.reset() # Ensure clean state (redundant after init, but safe)
        logger.info(f"Iteration {iteration}: Initialized HeadsUpPoker game with seed {iter_seed}. Log: {iteration_log_path}")

        # --- Create Game Adapter ---
        adapter = GameAdapter(game_type="poker")

        # --- Create Agent Functions ---
        # Pass model data and adapter to create the agent function
        agent1_func = self._create_poker_agent(models[player1_id], adapter)
        agent2_func = self._create_poker_agent(models[player2_id], adapter)

        # --- Run the Non-Interactive Game (set of hands) ---
        num_hands = self.config.get("num_hands", 10) # Get number of hands from config
        logger.info(f"Iteration {iteration}: Starting game with {num_hands} independent hands. {player1_id} ({models[player1_id]['name']}) vs {player2_id} ({models[player2_id]['name']})")

        # Call the modified run_non_interactive_game
        result_dict = run_non_interactive_game(
            game=game,
            player1_agent=agent1_func,
            player2_agent=agent2_func,
            num_hands=num_hands
        )

        # --- Augment Result ---
        # Add metadata specific to this iteration run
        result_dict["iteration"] = iteration
        result_dict["player_models"] = self._get_player_model_names(models)
        result_dict["random_seed_used"] = iter_seed
        # Game config is already included in the result_dict by run_non_interactive_game

        logger.info(f"Iteration {iteration} finished. Hands Completed: {result_dict.get('hands_played', 0)}/{num_hands}. Log: {iteration_log_path}")
        # logger.debug(f"Iteration {iteration} aggregate result: {result_dict.get('aggregate_results')}")

        return result_dict


    def run(self) -> Dict[str, Any]:
        """ Execute the entire experiment over all iterations. """
        experiment_name = self.config.get('name', 'unknown_experiment')
        logger.info(f"Starting experiment: {experiment_name}")
        self.start_time = time.time()

        all_iteration_results = []
        experiment_metadata = {
             "name": experiment_name,
             "description": self.config.get("description", ""),
             "config": self.config # Store the config used for the run
        }

        try:
            # --- Initialize Models (once for the experiment) ---
            games, models = self._initialize_experiment() # games dict is unused here

            # Run iterations
            iterations = self.config.get("iterations", 1)
            for i in range(iterations):
                iter_num = i + 1
                logger.info(f"--- Running Iteration {iter_num}/{iterations} ---")
                try:
                    # --- Call _run_iteration with models ---
                    # games dict is passed but not used by this runner's implementation
                    iteration_result = self._run_iteration(iter_num, games, models)
                    all_iteration_results.append(iteration_result)
                    # Log summary of the iteration's aggregate result
                    agg_res = iteration_result.get("aggregate_results", {})
                    p1_avg = agg_res.get(f"{list(models.keys())[0]}_avg_net_winnings_per_hand", "N/A")
                    p2_avg = agg_res.get(f"{list(models.keys())[1]}_avg_net_winnings_per_hand", "N/A")
                    logger.info(f"Iteration {iter_num} Aggregate Avg Winnings/Hand: P1={p1_avg:.2f}, P2={p2_avg:.2f}")

                except Exception as e:
                    logger.exception(f"Error during iteration {iter_num}: {e}")
                    all_iteration_results.append({
                        "iteration": iter_num,
                        "status": "error",
                        "error_message": str(e),
                        "player_models": self._get_player_model_names(models)
                    })

            self.end_time = time.time()
            duration = self.end_time - self.start_time
            logger.info(f"All {iterations} iterations completed in {duration:.2f} seconds.")

            # --- Analyze Results ---
            logger.info("Analyzing experiment results...")
            # Use the analyze_results method which should use MetricsTracker
            self.analysis = self.analyze_results(all_iteration_results)
            # Add overall experiment metadata to the analysis report
            self.analysis["experiment_metadata"] = {
                **experiment_metadata,
                "total_iterations_run": len(all_iteration_results),
                "duration_seconds": duration,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)),
            }
            logger.info("Analysis complete.")
            # logger.debug(f"Analysis result structure: {json.dumps(self.analysis, indent=2, default=str)}")

            # --- Save Results ---
            logger.info("Saving experiment results and analysis...")
            # Use save_results method which delegates to ResultStorage
            save_path = self.save_results(all_iteration_results, self.analysis)
            logger.info(f"Results saved to: {save_path}")


            return self.analysis

        except Exception as e:
            self.end_time = time.time()
            logger.exception(f"Experiment '{experiment_name}' failed critically: {e}")
            # Attempt to save partial results if any iterations completed
            if all_iteration_results:
                 try:
                      duration = self.end_time - self.start_time if self.start_time else None
                      partial_analysis = {"error": f"Experiment failed critically: {e}", "status": "failed"}
                      # Ensure metadata exists even for partial save
                      metadata_for_save = self.analysis.get("experiment_metadata", experiment_metadata) # Use existing analysis if available
                      metadata_for_save["status"] = "failed"
                      metadata_for_save["duration_seconds"] = duration
                      partial_save_path = self.result_storage.store(all_iteration_results, partial_analysis, metadata_for_save)
                      logger.info(f"Partial results saved due to critical failure: {partial_save_path}")
                 except Exception as save_err:
                      logger.error(f"Could not save partial results after critical failure: {save_err}")

            self.analysis = {"error": f"Experiment failed critically: {e}", "partial_results_logged": bool(all_iteration_results)}
            return self.analysis


    # --- Helper Methods ---

    def _initialize_models(self) -> Dict[str, Dict[str, Any]]:
        """ Initialize LLM clients based on configuration. """
        models_data = {}
        model_configs = self.config.get("models", [])
        if not model_configs or len(model_configs) < 1:
            logger.error("No models specified in configuration.")
            raise ValueError("Experiment configuration must include at least one model.")

        # Ensure exactly two models for heads-up play
        if len(model_configs) == 1:
             logger.warning("Only one model specified. It will play against itself.")
             # Deep copy is important if the model needs independent state later (though client is stateless)
             model_configs = [model_configs[0], copy.deepcopy(model_configs[0])]
        elif len(model_configs) > 2:
             logger.warning(f"More than two models specified ({len(model_configs)}). Using the first two.")
             model_configs = model_configs[:2]


        for i, model_config in enumerate(model_configs):
             # Use descriptive IDs like Player1, Player2
             player_id = f"Player{i+1}"

             logger.info(f"Initializing model for {player_id}: {model_config.get('name')}")
             try:
                 # Create LLMClient instance
                 client = LLMClient(
                     provider=model_config.get("provider"),
                     model=model_config.get("name"),
                     max_tokens=model_config.get("max_tokens", 150), # Default from original code
                     temperature=model_config.get("temperature", 0.7), # Default from models.yaml
                     max_retries=model_config.get("max_retries", 3),
                     timeout=model_config.get("timeout", 60)
                 )
                 # Store client and config under the player ID
                 models_data[player_id] = {
                     "id": player_id, # Store the ID itself
                     "config": model_config,
                     "client": client,
                     "name": model_config.get("name") # Store model name for logging
                 }
             except Exception as e:
                 logger.error(f"Failed to initialize model {model_config.get('name')} for {player_id}: {e}")
                 raise # Propagate error

        if len(models_data) != 2:
             logger.error("Could not initialize exactly two models for the heads-up game.")
             raise ValueError("Failed to initialize the required two models.")

        return models_data

    def _get_player_model_names(self, models: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """ Helper to get mapping of PlayerID -> ModelName used in the iteration results."""
        return {player_id: m_data.get("name", "unknown_model") for player_id, m_data in models.items()}

    def _create_poker_agent(self, model_data: Dict[str, Any], adapter: GameAdapter):
        """
        Creates a poker agent function closure that captures the LLM client and adapter.

        Args:
            model_data: Dictionary containing the 'client', 'id', and 'name' for the model.
            adapter: The GameAdapter instance to use for prompt/response handling.

        Returns:
            A function that takes game_state and valid_actions and returns a chosen action.
        """
        llm_client = model_data["client"]
        model_id = model_data["id"] # Player1 or Player2
        model_name = model_data["name"] # e.g., gpt-4o-mini
        # System prompt is static for the game type, get it once
        system_prompt = adapter.prepare_system_prompt()

        def agent_fn(game_state_dict: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
            """
            The actual function called by the game loop (run_non_interactive_game).
            It prepares the prompt, calls the LLM, parses the response, and returns an action.
            """
            # logger.debug(f"Agent {model_id} ({model_name}) invoked. Stage: {game_state_dict.get('stage')}, Turn: {game_state_dict.get('active_player')}")

            # --- Prepare Prompt ---
            # Use a temporary adapter instance or pass necessary info?
            # Pass required info directly to avoid state issues with adapter instance.
            prompt = ""
            try:
                # We need the full game history, which is part of the state dict now
                game_history_list = game_state_dict.get("history", [])
                # Use adapter's method to prepare variables, passing only necessary parts of state
                prompt_vars = adapter._prepare_poker_variables(
                     state=game_state_dict, # Pass the state dict received
                     valid_actions=valid_actions,
                     player_id=model_id,
                     game_history=game_history_list # Pass history from state
                )
                prompt = render_template(adapter.prompt_template, prompt_vars)
            except Exception as prompt_err:
                 logger.exception(f"Agent {model_id}: Error preparing prompt variables: {prompt_err}. Using fallback.")
                 # Ensure fallback uses the *correct* valid_actions list
                 return adapter.response_parser.get_fallback_action(valid_actions)

            # --- Call LLM ---
            start_time = time.time()
            raw_response_dict = llm_client.call_llm(
                developer_message=( # Internal context for the LLM call
                    f"You are {model_id} playing heads-up Texas Hold'em poker against one opponent. "
                    f"Your model name is {model_name}. Analyze the game state and history provided in the user message. "
                    "Choose the best action from the valid options presented. "
                    "Your reply MUST be ONLY the JSON object representing your chosen action, with no other text."
                ),
                user_message=prompt, # The detailed game state and request for action
                system_message=system_prompt # The overall persona/instructions
            )
            duration = time.time() - start_time

            # --- Parse Response ---
            response_text = parse_response_text(raw_response_dict)
            action = None
            log_action = None # Action to log (might be fallback)

            if not response_text:
                logger.warning(f"Agent {model_id}: LLM call failed or returned no text content (Duration: {duration:.2f}s). Using fallback.")
                action = adapter.response_parser.get_fallback_action(valid_actions)
                log_action = action # Log the fallback action
                log_llm_call( # Log the failed call attempt
                    model_id=f"{model_id}({model_name})",
                    prompt=prompt,
                    response_text="LLM Call Failed/Empty",
                    parsed_action=log_action # Log the fallback used
                )
            else:
                # Parse and validate the response using the adapter
                action = adapter.parse_and_validate_response(response_text, valid_actions)
                log_action = action # Log the action chosen (could be fallback if parsing/validation failed)
                # Log the successful call, response, and parsed action
                log_llm_call(
                     model_id=f"{model_id}({model_name})",
                     prompt=prompt,
                     response_text=response_text,
                     parsed_action=log_action
                )

            logger.debug(f"Agent {model_id} ({model_name}) chose action: {action} (LLM Duration: {duration:.2f}s)")
            return action # Return the validated (or fallback) action

        return agent_fn # Return the closure

    # --- Override analyze_results and save_results to use Tracker/Storage ---
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ Analyze experiment results using MetricsTracker. """
        logger.info("Using MetricsTracker to analyze results.")
        # Pass the list of iteration result dictionaries to the tracker
        # The tracker needs to be updated to handle the new result structure (with aggregate_results)
        return self.metrics_tracker.analyze(results)

    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """ Save results and analysis using ResultStorage. """
        logger.info("Using ResultStorage to save results.")
        # Extract metadata from the analysis dict where it should now reside
        metadata = analysis.get("experiment_metadata", {})
        if not metadata:
             logger.warning("Experiment metadata not found in analysis dict for saving. Creating basic metadata.")
             metadata = {
                 "name": self.config.get("name", "unknown"),
                 "description": self.config.get("description", ""),
                 "config": self.config,
                 "iterations_run": len(results),
                 "duration_seconds": self.end_time - self.start_time if self.end_time and self.start_time else None
             }

        # Pass results list, analysis dict, and metadata to storage
        filepath = self.result_storage.store(results, analysis, metadata)
        return filepath if filepath else ""
