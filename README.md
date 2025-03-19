# GameBench - LLM Game Benchmarking Framework

GameBench is a framework for benchmarking Large Language Models (LLMs) on various games. It provides a standardized way to evaluate LLM performance in strategic game environments.

## Overview

The framework supports:
- Texas Hold'em Poker (heads-up format)
- Multiple LLM providers (OpenAI, Anthropic, OpenRouter)
- Configurable experiment parameters
- Metrics collection and analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gamebench.git
cd gamebench
```

2. Set up API keys:
```bash
# For OpenAI models
export OPENAI_GAMEBENCH_KEY=your_openai_api_key

# For Anthropic models
export CLAUDE_GAMEBENCH_KEY=your_claude_api_key

# For OpenRouter models
export OPENROUTER_GAMEBENCH_KEY=your_openrouter_api_key
```

## Usage

### Running an Experiment

To run an experiment, use the `run_experiment.py` script:

```bash
python scripts/run_experiment.py --config experiments/config/examples/poker_benchmark.yaml
```

Options:
- `--config` or `-c`: Path to experiment configuration file (required)
- `--output` or `-o`: Output directory for results (optional, overrides config)
- `--iterations` or `-i`: Number of iterations to run (optional, overrides config)
- `--verbose` or `-v`: Enable verbose logging (optional)

### Configuration Files

Experiment configurations are defined in YAML files. See `experiments/config/examples/poker_benchmark.yaml` for an example.

Key configuration sections:
- `game`: Game type and parameters
- `models`: List of models to benchmark
- `iterations`: Number of experiment iterations
- `metrics`: Metrics to calculate
- `output`: Output format and directory

## Project Structure

- `game_engines/`: Game implementations
  - `base_game.py`: Abstract base class for all games
  - `heads_up_poker.py`: Poker game implementation
- `model_orchestrator/`: LLM client and game adapter
  - `llm_client.py`: Client for different LLM providers
  - `game_adapter.py`: Adapter between games and LLMs
  - `prompt_templates/`: Templates for prompting LLMs
  - `response_parsers/`: Parsers for LLM responses
- `experiments/`: Experiment configurations and runners
  - `config/`: Configuration schemas and loaders
  - `runners/`: Experiment execution logic
  - `results/`: Result storage and metrics
- `scripts/`: Utility scripts
  - `run_experiment.py`: Main script for running experiments

## Adding Support for New Games

To add a new game:

1. Create a new game implementation in `game_engines/`
2. Add prompt templates in `model_orchestrator/prompt_templates/templates/`
3. Add a response parser in `model_orchestrator/response_parsers/`
4. Update the game adapter in `model_orchestrator/game_adapter.py` to support the new game
5. Add default config in `experiments/config/default_configs/`

## License

[MIT License](LICENSE)