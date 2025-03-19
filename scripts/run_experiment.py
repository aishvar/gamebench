#!/usr/bin/env python3
"""
Script for running LLM game benchmarking experiments.
"""

import os
import sys
import argparse
import logging
import json
import time
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import load_config, validate_config
from experiments.runners import SequentialRunner
from experiments.results import ResultStorage, MetricsTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM game benchmarking experiments")
    
    parser.add_argument(
        "--config", 
        "-c", 
        type=str, 
        required=True, 
        help="Path to experiment configuration file"
    )
    
    parser.add_argument(
        "--output", 
        "-o", 
        type=str, 
        help="Output directory for results (overrides config)"
    )
    
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--iterations", 
        "-i", 
        type=int, 
        help="Number of iterations to run (overrides config)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load and validate configuration
    try:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Override configuration if specified in arguments
        if args.output:
            if "output" not in config:
                config["output"] = {}
            config["output"]["directory"] = args.output
            logger.info(f"Overriding output directory: {args.output}")
            
        if args.iterations:
            config["iterations"] = args.iterations
            logger.info(f"Overriding iterations: {args.iterations}")
            
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Run the experiment
    try:
        logger.info(f"Starting experiment: {config.get('name', 'unknown')}")
        start_time = time.time()
        
        # Initialize and run the experiment
        runner = SequentialRunner(config)
        analysis = runner.run()
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Experiment completed in {duration:.2f} seconds")
        
        # Log summary of results
        if "metrics" in analysis:
            logger.info("=== Experiment Results ===")
            for metric_name, metric_value in analysis["metrics"].items():
                logger.info(f"{metric_name}: {metric_value}")
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()