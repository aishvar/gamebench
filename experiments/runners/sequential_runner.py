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
from game_engines.heads_up_poker import HeadsUpPoker, run_non_interactive_game
from model_orchestrator.utils import init_game_log, log_llm_call, log_event_to_game_file, parse_response_text
# --- Ensure this import is present ---
from model_orchestrator.prompt_templates import render_template

from model_orchestrator.llm_client import LLMClient
from model_orchestrator.game_adapter import GameAdapter
from experiments.runners.base_runner import ExperimentRunner
from experiments.results import ResultStorage, MetricsTracker

class SequentialRunner(ExperimentRunner):
    """
    Sequential experiment runner for turn-based games.
    Runs experiments one at a time in a single thread.
    """

    def __init__(self, config: Dict[str, Any]):
        """ Initialize the sequential experiment runner. """
        super().__init__(config)
        self.metrics_tracker = MetricsTracker(config.get("metrics", []))
        self.result_storage = ResultStorage(config.get("output", {}))

        log_dir = config.get("output", {}).get("directory", "./results")
        log_subdir = os.path.join(log_dir, "game_logs")
        os.makedirs(log_subdir, exist_ok=True)
        init_game_log(log_subdir)
        logger.info(f"Game logs will be saved in: {log_subdir}")

    # --- Implementation of Abstract Methods ---

    def _initialize_experiment(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Initialize models for the experiment.
        Games are initialized per iteration in this runner.

        Returns:
            Tuple of (games, models) dictionaries. Games dict is empty here.
        """
        logger.info("Initializing models for the experiment.")
        models = self._initialize_models()
        logger.info(f"Initialized {len(models)} models.")
        # No persistent game instances needed across iterations for this runner
        games = {}
        return games, models

    # Corrected signature to match base class
    def _run_iteration(self, iteration: int, games: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single iteration (a full game of multiple hands) of the experiment.

        Args:
            iteration: Iteration number (for logging).
            games: Dictionary of game instances (unused in this runner, but required by signature).
            models: Dictionary of initialized model data (client, config).

        Returns:
            Results dictionary for this iteration (game outcome).
        """
        game_config = self.config.get("game", {})
        game_type = game_config.get("type", "poker")

        if game_type != "poker":
            raise ValueError(f"Unsupported game type in SequentialRunner: {game_type}")

        # --- Initialize Game Instance for this Iteration ---
        base_seed = game_config.get("random_seed")
        iter_seed = hash((base_seed, iteration)) if base_seed is not None else None

        player_ids = list(models.keys())
        player1_id = player_ids[0]
        player2_id = player_ids[1]

        # Create the game instance specifically for this iteration
        game = HeadsUpPoker(
            player1_name=player1_id,
            player2_name=player2_id,
            starting_stack=game_config.get("starting_stack", 1000),
            small_blind=game_config.get("small_blind", 10),
            big_blind=game_config.get("big_blind", 20),
            random_seed=iter_seed
        )
        game.reset() # Ensure clean state
        logger.info(f"Iteration {iteration}: Initialized HeadsUpPoker game with seed {iter_seed}.")

        # --- Create Game Adapter ---
        adapter = GameAdapter(game_type="poker")

        # --- Create Agent Functions ---
        agent1_func = self._create_poker_agent(models[player1_id], adapter)
        agent2_func = self._create_poker_agent(models[player2_id], adapter)

        # --- Run the Non-Interactive Game ---
        num_hands = self.config.get("num_hands", 10)
        logger.info(f"Iteration {iteration}: Starting game with {num_hands} hands. {player1_id} ({models[player1_id]['name']}) vs {player2_id} ({models[player2_id]['name']})")

        result = run_non_interactive_game(
            game=game,
            player1_agent=agent1_func,
            player2_agent=agent2_func,
            num_hands=num_hands
        )

        # --- Augment Result ---
        result["iteration"] = iteration
        result["player_models"] = self._get_player_model_names(models)
        result["game_config"] = game_config

        return result


    def run(self) -> Dict[str, Any]:
        """ Execute the experiment. """
        experiment_name = self.config.get('name', 'unknown_experiment')
        logger.info(f"Starting experiment: {experiment_name}")
        self.start_time = time.time()

        all_iteration_results = []
        experiment_metadata = {
             "name": experiment_name,
             "description": self.config.get("description", ""),
             "config": self.config
        }

        try:
            # --- Call _initialize_experiment ---
            games, models = self._initialize_experiment() # Get initialized models

            # Run iterations
            iterations = self.config.get("iterations", 1)
            for i in range(iterations):
                iter_num = i + 1
                logger.info(f"--- Running Iteration {iter_num}/{iterations} ---")
                log_event_to_game_file(f"\n===== ITERATION {iter_num} / {iterations} =====\n")
                try:
                    # --- Call _run_iteration with games and models ---
                    iteration_result = self._run_iteration(iter_num, games, models)
                    all_iteration_results.append(iteration_result)
                    logger.info(f"Iteration {iter_num} finished. Winner: {iteration_result.get('winner')}, Hands: {iteration_result.get('hands_played')}")
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
            logger.info(f"All iterations completed in {duration:.2f} seconds.")

            logger.info("Analyzing experiment results...")
            self.analysis = self.metrics_tracker.analyze(all_iteration_results)
            self.analysis["experiment_metadata"] = {
                **experiment_metadata,
                "total_iterations_run": len(all_iteration_results),
                "duration_seconds": duration,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)),
            }
            logger.info("Analysis complete.")
            logger.debug(f"Analysis result: {json.dumps(self.analysis, indent=2, default=str)}")

            logger.info("Saving experiment results and analysis...")
            # Use save_results method which now delegates to ResultStorage
            self.save_results(all_iteration_results, self.analysis)

            return self.analysis

        except Exception as e:
            self.end_time = time.time()
            logger.exception(f"Experiment '{experiment_name}' failed critically: {e}")
            if all_iteration_results:
                 try:
                      duration = self.end_time - self.start_time if self.start_time else None
                      partial_analysis = {"error": f"Experiment failed: {e}"}
                      # Ensure metadata exists even for partial save
                      metadata_for_save = self.analysis.get("experiment_metadata", experiment_metadata)
                      metadata_for_save["status"] = "failed"
                      metadata_for_save["duration_seconds"] = duration
                      self.result_storage.store(all_iteration_results, partial_analysis, metadata_for_save)
                      logger.info("Partial results saved due to critical failure.")
                 except Exception as save_err:
                      logger.error(f"Could not save partial results after critical failure: {save_err}")

            self.analysis = {"error": f"Experiment failed critically: {e}", "partial_results_logged": bool(all_iteration_results)}
            return self.analysis


    # --- Helper Methods (moved from old run/init logic) ---

    def _initialize_models(self) -> Dict[str, Dict[str, Any]]:
        """ Initialize LLM clients based on configuration. """
        models = {}
        model_configs = self.config.get("models", [])
        if not model_configs or len(model_configs) < 1:
            logger.error("No models specified in configuration.")
            raise ValueError("Experiment configuration must include at least one model.")

        if len(model_configs) == 1:
             logger.warning("Only one model specified. It will play against itself.")
             model_configs = [model_configs[0], copy.deepcopy(model_configs[0])]

        for i, model_config in enumerate(model_configs[:2]):
             player_name = f"Player{i+1}"
             model_id = player_name

             logger.info(f"Initializing model for {model_id}: {model_config.get('name')}")
             try:
                 client = LLMClient(
                     provider=model_config.get("provider"),
                     model=model_config.get("name"),
                     max_tokens=model_config.get("max_tokens", 150),
                     temperature=model_config.get("temperature", 0.5),
                     max_retries=model_config.get("max_retries", 3),
                     timeout=model_config.get("timeout", 60)
                 )
                 models[model_id] = {
                     "id": model_id,
                     "config": model_config,
                     "client": client,
                     "name": model_config.get("name")
                 }
             except Exception as e:
                 logger.error(f"Failed to initialize model {model_config.get('name')} for {model_id}: {e}")
                 raise

        if len(models) < 2:
             logger.error("Could not initialize two models for the heads-up game.")
             raise ValueError("Failed to initialize the required two models.")

        return models

    def _get_player_model_names(self, models: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """ Helper to get mapping of PlayerID -> ModelName """
        return {m_id: m_data.get("name", "unknown") for m_id, m_data in models.items()}

    def _create_poker_agent(self, model_data: Dict[str, Any], adapter: GameAdapter):
        """ Create a poker agent function closure. """
        llm_client = model_data["client"]
        model_id = model_data["id"]
        model_name = model_data["name"]
        system_prompt = adapter.prepare_system_prompt()

        def agent_fn(game_state_dict: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
            """ The actual function called by the game loop. """
            logger.debug(f"Agent {model_id} ({model_name}) invoked. Stage: {game_state_dict.get('stage')}")

            temp_prompt_adapter = GameAdapter(game_type="poker")
            try:
                game_history_list = game_state_dict.get("history", [])
                prompt_vars = temp_prompt_adapter._prepare_poker_variables(
                     state=game_state_dict,
                     valid_actions=valid_actions,
                     player_id=model_id,
                     game_history=game_history_list
                )
                # --- Use imported render_template ---
                prompt = render_template(temp_prompt_adapter.prompt_template, prompt_vars)
            except Exception as prompt_err:
                 logger.exception(f"Agent {model_id}: Error preparing prompt variables: {prompt_err}")
                 return adapter.response_parser.get_fallback_action(valid_actions)

            raw_response_dict = llm_client.call_llm(
                developer_message=(
                    f"You are {model_id} playing heads-up Texas Hold'em poker. "
                    "Analyze the state and choose the best action from the valid options. "
                    "Reply ONLY with the JSON for your chosen action."
                ),
                user_message=prompt,
                system_message=system_prompt
            )

            response_text = parse_response_text(raw_response_dict)
            log_llm_call(
                model_id=f"{model_id}({model_name})",
                prompt=prompt,
                response_text=response_text if response_text else "No response text",
                parsed_action=None
            )

            if not response_text:
                logger.warning(f"Agent {model_id}: LLM call failed or returned no text content. Using fallback.")
                action = adapter.response_parser.get_fallback_action(valid_actions)
                log_llm_call(
                    model_id=f"{model_id}({model_name})",
                    prompt=prompt,
                    response_text="LLM Call Failed/Empty",
                    parsed_action=action,
                    is_retry=True
                )
                return action

            action = adapter.parse_and_validate_response(response_text, valid_actions)
            log_llm_call(
                 model_id=f"{model_id}({model_name})",
                 prompt="See previous LLM Call entry",
                 response_text=response_text,
                 parsed_action=action
            )

            logger.debug(f"Agent {model_id} chose action: {action}")
            return action

        return agent_fn

    # --- Override analyze_results and save_results to use Tracker/Storage ---
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ Analyze experiment results using MetricsTracker. """
        logger.info("Using MetricsTracker to analyze results.")
        return self.metrics_tracker.analyze(results)

    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """ Save results and analysis using ResultStorage. """
        logger.info("Using ResultStorage to save results.")
        metadata = analysis.get("experiment_metadata", {})
        if not metadata:
             logger.warning("Experiment metadata not found in analysis dict for saving.")
             metadata = {
                 "name": self.config.get("name", "unknown"),
                 "iterations": len(results),
                 "duration_seconds": self.end_time - self.start_time if self.end_time and self.start_time else None
             }

        filepath = self.result_storage.store(results, analysis, metadata)
        return filepath if filepath else ""