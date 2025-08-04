import os
import sys
import logging
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import get_human_eval_subset
from src.llm import LLMHandler
from src.verification import run_unit_tests, run_smt_verification

# --- Configuration ---
N_PROBLEMS = 3  # Use a small number for a quick test run
MAX_REPAIR_ITERATIONS = 3
RESULTS_DIR = "results"
LOG_FILE = os.path.join(RESULTS_DIR, "log.txt")
RESULTS_CSV = os.path.join(RESULTS_DIR, "experiment_results.csv")

# --- Setup ---
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)

def run_experiment():
    """Main function to orchestrate the experiment."""
    
    logging.info("Starting VerifAI experiment...")
    
    # Initialize components
    try:
        llm = LLMHandler()
        problems = get_human_eval_subset(N_PROBLEMS)
    except Exception as e:
        logging.error(f"Failed to initialize components: {e}")
        return

    results_data = []

    for problem in tqdm(problems, desc="Processing Problems"):
        task_id = problem['task_id']
        prompt = problem['prompt']
        entry_point = problem['entry_point']
        canonical_test = problem['test']
        
        logging.info(f"--- Processing Problem: {task_id} ---")

        # --- 1. Zero-Shot Baseline ---
        logging.info(f"Running Zero-Shot for {task_id}")
        try:
            zero_shot_code = llm.generate_code(prompt)
            passed, _ = run_unit_tests(zero_shot_code, canonical_test)
            results_data.append({
                "task_id": task_id,
                "method": "Zero-Shot",
                "iteration": 0,
                "passed": passed,
                "code": zero_shot_code
            })
            logging.info(f"Zero-Shot result for {task_id}: {'Passed' if passed else 'Failed'}")
        except Exception as e:
            logging.error(f"Error in Zero-Shot for {task_id}: {e}")
            passed = False # Mark as failed if an error occurs

        # Start with the zero-shot code for repair methods
        current_code = zero_shot_code
        
        # --- 2. UT-Repair Baseline ---
        logging.info(f"Running UT-Repair for {task_id}")
        ut_passed = passed
        for i in range(MAX_REPAIR_ITERATIONS):
            if ut_passed:
                break # Already correct
            
            logging.info(f"UT-Repair iteration {i+1} for {task_id}")
            # Use the canonical test for feedback
            is_correct, feedback = run_unit_tests(current_code, canonical_test)
            if is_correct:
                ut_passed = True
                break

            try:
                current_code = llm.correct_code(prompt, current_code, feedback)
                ut_passed, _ = run_unit_tests(current_code, canonical_test)
            except Exception as e:
                logging.error(f"Error in UT-Repair for {task_id} at iteration {i+1}: {e}")
                ut_passed = False
                break # Stop trying if an error occurs
        
        results_data.append({
            "task_id": task_id,
            "method": "UT-Repair",
            "iteration": i + 1 if not ut_passed else i,
            "passed": ut_passed,
            "code": current_code
        })
        logging.info(f"UT-Repair final result for {task_id}: {'Passed' if ut_passed else 'Failed'}")

        # Reset to original zero-shot code for the next method
        current_code = zero_shot_code

        # --- 3. SMT-Repair (Proposed Method) ---
        logging.info(f"Running SMT-Repair for {task_id}")
        smt_passed = passed
        for i in range(MAX_REPAIR_ITERATIONS):
            if smt_passed:
                break

            logging.info(f"SMT-Repair iteration {i+1} for {task_id}")
            # Use SMT for feedback
            is_correct, feedback = run_smt_verification(current_code, entry_point)
            if is_correct:
                # If SMT passes, we still need to do a final check with canonical tests
                smt_passed, _ = run_unit_tests(current_code, canonical_test)
                break

            try:
                current_code = llm.correct_code(prompt, current_code, feedback)
                smt_passed, _ = run_unit_tests(current_code, canonical_test)
            except Exception as e:
                logging.error(f"Error in SMT-Repair for {task_id} at iteration {i+1}: {e}")
                smt_passed = False
                break

        results_data.append({
            "task_id": task_id,
            "method": "SMT-Repair",
            "iteration": i + 1 if not smt_passed else i,
            "passed": smt_passed,
            "code": current_code
        })
        logging.info(f"SMT-Repair final result for {task_id}: {'Passed' if smt_passed else 'Failed'}")

    # --- Save Results ---
    df = pd.DataFrame(results_data)
    df.to_csv(RESULTS_CSV, index=False)
    logging.info(f"Experiment results saved to {RESULTS_CSV}")

    # --- Generate Plots ---
    logging.info("Generating result plots...")
    try:
        # This will be a separate script
        import plot_results
        plot_results.main()
        logging.info("Plots generated successfully.")
    except Exception as e:
        logging.error(f"Failed to generate plots: {e}")

    logging.info("Experiment finished.")

if __name__ == "__main__":
    run_experiment()
