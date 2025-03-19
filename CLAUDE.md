# GameBench Development Guidelines

## Commands
- Run experiment: `python scripts/run_experiment.py`
- Run poker game: `python game_engines/heads_up_poker.py`
- Run tests (create these): `pytest tests/`
- Linting: `flake8 .`
- Type checking: `mypy .`

## Code Style
- **Imports**: Standard library first, then third-party, then local modules
- **Type Hints**: Add type annotations to function parameters and returns
- **Naming**: `snake_case` for variables/functions, `CamelCase` for classes
- **Docstrings**: Use multi-line docstrings for classes and functions
- **Error Handling**: Use try/except blocks with specific exceptions
- **Logging**: Use the logging module instead of print statements
- **Constants**: Use UPPERCASE for constants
- **Line Length**: Maximum 100 characters

## Project Structure
- Keep game engines in `game_engines/`
- Model integration code in `model_orchestrator/`
- Evaluation scripts in `evals/`
- Experimental scripts in `scripts/`
- LLM logs in `data/logs/raw_llm_logs/`