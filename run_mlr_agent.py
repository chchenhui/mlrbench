import os
import os.path as osp
import sys
import time
import json
import random
import shutil
import argparse
import logging
from datetime import datetime

from mlrbench.utils.utils import *
from mlrbench.llm.llm import *
from mlrbench.lmm.lmm import *
from mlrbench.agent.idea_generator import generate_idea_for_pipeline
from mlrbench.agent.lit_review import generate_lit_review_for_pipeline
from mlrbench.agent.proposal_generator import generate_proposal_for_pipeline
from mlrbench.agent.experiment_runner import run_experiment
from mlrbench.agent.paper_writer import write_paper_for_pipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run MLR Agent")
    parser.add_argument(
        "--model_name",
        type=str,
        default="claude-3-7-sonnet-20250219",
        help="Model to use for the MLR agent",
    )
    parser.add_argument(
        "--lit_engine_name",
        type=str,
        default="gpt-4o-search-preview-2025-03-11",
        help="Literature engine to use for the MLR agent",
    )
    parser.add_argument(
        "--coding_agent",
        type=str,
        default="claude_code",
        help="Coding agent to use for the MLR agent",
    )
    return parser.parse_args()


def run_mlr_agent(
    model_name, 
    lit_engine_name,
    coding_agent,
    task_path,
    ):
    """
    Run the MLR agent with logging and basic setup.
    """   
    # Check the environment
    logging.info("Checking environment...")
    if "claude" in model_name or "claude" in coding_agent:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise NotImplementedError("API_KEY is not set in environment variables.")
        if not shutil.which("claude"):
            raise NotImplementedError("Claude Code is not installed. Please install it first.")
    
    if "gpt" in model_name or "gpt" in lit_engine_name or "codex" in coding_agent:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise NotImplementedError("API_KEY is not set in environment variables.")
    
    logging.info(f"MLR Agent for {task_path} started.")
    logging.info(f"Using model: {model_name}")
    logging.info(f"Using literature engine: {lit_engine_name}")
    # Check if the task exists
    task_file = osp.join(task_path, "task.md") 
    if not os.path.exists(task_file):
        raise NotImplementedError(f"Task file {task_file} does not exist.")
    # Generate idea
    logging.info("Generating idea...")
    generate_idea_for_pipeline(model_name, task_path)
    logging.info("Idea generated.")
    # Generate literature review
    logging.info("Generating literature review...")
    generate_lit_review_for_pipeline(lit_engine_name, task_path)
    logging.info("Literature review generated.")
    # Generate proposal
    logging.info("Generating proposal...")
    generate_proposal_for_pipeline(model_name, task_path)
    logging.info("Proposal generated.")
    # Run the experiment
    logging.info("Running experiment...")
    run_experiment(coding_agent, task_path)
    logging.info("Experiment completed.")
    # Write the paper
    logging.info("Writing paper...")
    write_paper_for_pipeline(model_name, task_path)
    logging.info("Paper written.")
    logging.info(f"MLR Agent for {task_path} finished!")

    
def main():
    args = parse_arguments()
    model_name = args.model_name
    # model_name="claude-3-7-sonnet-20250219"
    # model_name = "google/gemini-2.5-pro-preview"
    lit_engine_name = args.lit_engine_name
    coding_agent = args.coding_agent
    if "claude" in model_name:
        simple_name = "claude"
    elif "gemini" in model_name:
        simple_name = "gemini"
        if "gemini" in coding_agent:
            simple_name = "gemini-cli"
    elif "o4-mini" in model_name:
        simple_name = "o4-mini"
        if "codex" in coding_agent:
            simple_name = "codex"
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"mlr_agent_{simple_name}.log"),
            logging.StreamHandler()
        ]
    )
    task_folder = f"end2end_{simple_name}"
    if not os.path.exists(task_folder):
        raise NotImplementedError(f"Task folder {task_folder} does not exist.")
    tasklist = get_tasklist(task_folder)
    logging.info(f"Task list: {tasklist}")
    logging.info("Starting MLR agent...")
    for task_name in tasklist:
        task_path = osp.join(task_folder, task_name)
        # Run the MLR agent for each task
        logging.info(f"Running MLR agent for task: {task_name}")
        run_mlr_agent(model_name=model_name, 
                      lit_engine_name=lit_engine_name,
                      coding_agent=coding_agent,
                      task_path=task_path)
    logging.info("All tasks completed.")
    

if __name__ == "__main__":
    main()