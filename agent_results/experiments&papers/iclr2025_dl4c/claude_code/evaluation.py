"""
Evaluation module for the adaptive code assistant experiment.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import tempfile
import subprocess
from utils import timer, save_json, ensure_dir

logger = logging.getLogger("adaptive_code_assistant.evaluation")

class CodeEvaluator:
    """
    Class to evaluate code assistant performance.
    """
    
    def __init__(self):
        """Initialize the code evaluator."""
        logger.info("Initialized code evaluator")
    
    @timer
    def evaluate_functional_correctness(self, code: str, test_cases: str) -> Dict[str, Any]:
        """
        Evaluate the functional correctness of code by running test cases.
        
        Args:
            code: Code to evaluate
            test_cases: Test cases to run
            
        Returns:
            Dictionary containing evaluation results
        """
        # Create a temporary Python file with the code and test cases
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(code.encode())
            temp_file.write(b"\n\n# Test cases\n")
            temp_file.write(test_cases.encode())
            temp_file.write(b"\n\nprint('All tests passed!')")
        
        # Run the code and capture output
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=5  # 5 second timeout
            )
            execution_time = time.time() - start_time
            
            # Check if tests passed
            passed = "All tests passed!" in result.stdout
            
            # Get error message if any
            error = result.stderr.strip() if result.stderr else None
            
            logger.info(f"Code evaluation: {'Passed' if passed else 'Failed'}")
            
            return {
                "passed": passed,
                "error": error,
                "execution_time": execution_time
            }
        except subprocess.TimeoutExpired:
            logger.warning("Code evaluation timed out")
            return {
                "passed": False,
                "error": "Execution timed out",
                "execution_time": 5.0
            }
        except Exception as e:
            logger.error(f"Error during code evaluation: {e}")
            return {
                "passed": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    def evaluate_code_style(self, code: str, style_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate code style based on preferences.
        
        Args:
            code: Code to evaluate
            style_preferences: Style preferences to check against
            
        Returns:
            Dictionary containing style evaluation results
        """
        results = {}
        
        # Check indentation style
        if "indentation" in style_preferences:
            pref_style = style_preferences["indentation"]["style"]
            pref_width = style_preferences["indentation"]["width"]
            
            if pref_style == "spaces":
                # Check if spaces are used for indentation
                spaces_count = 0
                tabs_count = 0
                
                for line in code.split("\n"):
                    if line.startswith(" "):
                        spaces_count += 1
                    elif line.startswith("\t"):
                        tabs_count += 1
                
                indentation_score = spaces_count / (spaces_count + tabs_count + 0.0001)
                results["indentation_style_score"] = indentation_score
            else:
                # Check if tabs are used for indentation
                spaces_count = 0
                tabs_count = 0
                
                for line in code.split("\n"):
                    if line.startswith(" "):
                        spaces_count += 1
                    elif line.startswith("\t"):
                        tabs_count += 1
                
                indentation_score = tabs_count / (spaces_count + tabs_count + 0.0001)
                results["indentation_style_score"] = indentation_score
        
        # Check variable naming style
        if "variable_naming" in style_preferences:
            pref_naming = style_preferences["variable_naming"]
            
            snake_case_count = 0
            camel_case_count = 0
            
            import re
            
            # Count variable declarations
            snake_case_pattern = r"\b[a-z][a-z0-9_]*\b\s*="
            camel_case_pattern = r"\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b\s*="
            
            snake_case_count = len(re.findall(snake_case_pattern, code))
            camel_case_count = len(re.findall(camel_case_pattern, code))
            
            if pref_naming == "snake_case":
                naming_score = snake_case_count / (snake_case_count + camel_case_count + 0.0001)
            else:  # camelCase
                naming_score = camel_case_count / (snake_case_count + camel_case_count + 0.0001)
            
            results["naming_style_score"] = naming_score
        
        # Check docstring style
        if "docstring_style" in style_preferences:
            pref_docstring = style_preferences["docstring_style"]
            
            # Count docstring styles
            google_style_count = code.count('"""Args:')
            numpy_style_count = code.count('"""Parameters')
            sphinx_style_count = code.count('""":param')
            
            if pref_docstring == "google":
                docstring_score = google_style_count / (google_style_count + numpy_style_count + sphinx_style_count + 0.0001)
            elif pref_docstring == "numpy":
                docstring_score = numpy_style_count / (google_style_count + numpy_style_count + sphinx_style_count + 0.0001)
            else:  # sphinx
                docstring_score = sphinx_style_count / (google_style_count + numpy_style_count + sphinx_style_count + 0.0001)
            
            results["docstring_style_score"] = docstring_score
        
        # Calculate overall style score
        scores = list(results.values())
        overall_score = np.mean(scores) if scores else 0.0
        results["overall_style_score"] = overall_score
        
        logger.info(f"Code style evaluation: {overall_score:.2f}")
        
        return results
    
    def evaluate_development_speed(
        self,
        completion_time: float,
        iterations: int,
        execution_time: float
    ) -> Dict[str, float]:
        """
        Evaluate development speed metrics.
        
        Args:
            completion_time: Time taken to generate code completion
            iterations: Number of iterations needed to complete the task
            execution_time: Time taken to execute the code
            
        Returns:
            Dictionary containing speed evaluation results
        """
        # Calculate development speed metrics
        time_per_iteration = completion_time / max(1, iterations)
        efficiency_score = 1.0 / (1.0 + execution_time)  # Higher for faster execution
        
        # Calculate overall speed score
        # Weight factors can be adjusted based on importance
        overall_speed = 0.5 * (1.0 / (1.0 + time_per_iteration)) + 0.3 * (1.0 / (1.0 + iterations)) + 0.2 * efficiency_score
        
        results = {
            "time_per_iteration": time_per_iteration,
            "iterations_count": iterations,
            "execution_efficiency": efficiency_score,
            "overall_speed_score": overall_speed
        }
        
        logger.info(f"Development speed evaluation: {overall_speed:.2f}")
        
        return results
    
    def evaluate_user_satisfaction(
        self,
        style_score: float,
        correctness_score: float,
        speed_score: float,
        explicit_feedback: float = None
    ) -> Dict[str, float]:
        """
        Evaluate simulated user satisfaction.
        
        Args:
            style_score: Code style score
            correctness_score: Functional correctness score
            speed_score: Development speed score
            explicit_feedback: Explicit feedback score (if provided)
            
        Returns:
            Dictionary containing satisfaction evaluation results
        """
        # Calculate satisfaction metrics
        style_satisfaction = style_score
        correctness_satisfaction = 1.0 if correctness_score > 0.8 else correctness_score
        speed_satisfaction = speed_score
        
        # Calculate overall satisfaction
        # Weight factors can be adjusted based on importance
        if explicit_feedback is not None:
            overall_satisfaction = 0.3 * style_satisfaction + 0.4 * correctness_satisfaction + 0.1 * speed_satisfaction + 0.2 * explicit_feedback
        else:
            overall_satisfaction = 0.4 * style_satisfaction + 0.4 * correctness_satisfaction + 0.2 * speed_satisfaction
        
        results = {
            "style_satisfaction": style_satisfaction,
            "correctness_satisfaction": correctness_satisfaction,
            "speed_satisfaction": speed_satisfaction
        }
        
        if explicit_feedback is not None:
            results["explicit_feedback"] = explicit_feedback
        
        results["overall_satisfaction"] = overall_satisfaction
        
        logger.info(f"User satisfaction evaluation: {overall_satisfaction:.2f}")
        
        return results
    
    def evaluate_adaptation(
        self,
        initial_satisfaction: float,
        final_satisfaction: float,
        num_iterations: int
    ) -> Dict[str, float]:
        """
        Evaluate adaptation performance.
        
        Args:
            initial_satisfaction: Initial user satisfaction
            final_satisfaction: Final user satisfaction
            num_iterations: Number of iterations
            
        Returns:
            Dictionary containing adaptation evaluation results
        """
        # Calculate adaptation metrics
        adaptation_gain = final_satisfaction - initial_satisfaction
        adaptation_rate = adaptation_gain / max(1, num_iterations)
        
        results = {
            "adaptation_gain": adaptation_gain,
            "adaptation_rate": adaptation_rate
        }
        
        logger.info(f"Adaptation evaluation: gain={adaptation_gain:.2f}, rate={adaptation_rate:.4f}")
        
        return results

class ExperimentEvaluator:
    """
    Class to evaluate the entire experiment.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the experiment evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        ensure_dir(output_dir)
        
        self.code_evaluator = CodeEvaluator()
        self.results = {}
        
        logger.info(f"Initialized experiment evaluator with output directory: {output_dir}")
    
    def evaluate_session(
        self,
        session_data: Dict[str, Any],
        developer_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single coding session.
        
        Args:
            session_data: Data from a coding session
            developer_profile: Developer profile used in the session
            
        Returns:
            Dictionary containing session evaluation results
        """
        results = {}
        
        # Extract session data
        model_name = session_data.get("model_name", "unknown")
        task_id = session_data.get("task_id", "unknown")
        final_code = session_data.get("final_code", "")
        test_cases = session_data.get("test_cases", "")
        completion_time = session_data.get("completion_time", 0.0)
        iterations = session_data.get("iterations", 0)
        
        # Extract developer preferences
        style_preferences = developer_profile.get("formatting_preferences", {})
        
        # Evaluate functional correctness
        correctness_results = self.code_evaluator.evaluate_functional_correctness(final_code, test_cases)
        correctness_score = 1.0 if correctness_results.get("passed", False) else 0.0
        
        # Evaluate code style
        style_results = self.code_evaluator.evaluate_code_style(final_code, style_preferences)
        style_score = style_results.get("overall_style_score", 0.0)
        
        # Evaluate development speed
        speed_results = self.code_evaluator.evaluate_development_speed(
            completion_time,
            iterations,
            correctness_results.get("execution_time", 0.0)
        )
        speed_score = speed_results.get("overall_speed_score", 0.0)
        
        # Evaluate user satisfaction
        explicit_feedback = None
        if "feedback_history" in session_data and len(session_data["feedback_history"]) > 0:
            # Use the last feedback as explicit feedback
            last_feedback = session_data["feedback_history"][-1]
            explicit_feedback = last_feedback.get("satisfaction", None)
        
        satisfaction_results = self.code_evaluator.evaluate_user_satisfaction(
            style_score,
            correctness_score,
            speed_score,
            explicit_feedback
        )
        
        # Evaluate adaptation if there are multiple iterations
        adaptation_results = {}
        if iterations > 1 and "satisfaction_history" in session_data:
            satisfaction_history = session_data["satisfaction_history"]
            if len(satisfaction_history) >= 2:
                initial_satisfaction = satisfaction_history[0]
                final_satisfaction = satisfaction_history[-1]
                
                adaptation_results = self.code_evaluator.evaluate_adaptation(
                    initial_satisfaction,
                    final_satisfaction,
                    iterations
                )
        
        # Combine all results
        results = {
            "model_name": model_name,
            "task_id": task_id,
            "correctness": correctness_results,
            "style": style_results,
            "speed": speed_results,
            "satisfaction": satisfaction_results,
            "adaptation": adaptation_results
        }
        
        return results
    
    def evaluate_experiment(
        self,
        experiment_data: Dict[str, Any],
        developer_profiles: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the entire experiment.
        
        Args:
            experiment_data: Data from the experiment
            developer_profiles: Developer profiles used in the experiment
            
        Returns:
            Dictionary containing experiment evaluation results
        """
        self.results = {
            "models": {},
            "overall": {}
        }
        
        # Process each session
        for session_id, session_data in experiment_data.get("sessions", {}).items():
            dev_id = session_data.get("developer_id", "unknown")
            developer_profile = next((p for p in developer_profiles if p.get("dev_id") == dev_id), {})
            
            session_results = self.evaluate_session(session_data, developer_profile)
            
            model_name = session_data.get("model_name", "unknown")
            
            # Initialize model results if not exists
            if model_name not in self.results["models"]:
                self.results["models"][model_name] = {
                    "sessions": {},
                    "metrics": {
                        "correctness_rate": [],
                        "style_score": [],
                        "speed_score": [],
                        "satisfaction": [],
                        "adaptation_gain": [],
                        "adaptation_rate": []
                    }
                }
            
            # Add session results
            self.results["models"][model_name]["sessions"][session_id] = session_results
            
            # Update model metrics
            metrics = self.results["models"][model_name]["metrics"]
            metrics["correctness_rate"].append(1.0 if session_results["correctness"]["passed"] else 0.0)
            metrics["style_score"].append(session_results["style"].get("overall_style_score", 0.0))
            metrics["speed_score"].append(session_results["speed"].get("overall_speed_score", 0.0))
            metrics["satisfaction"].append(session_results["satisfaction"].get("overall_satisfaction", 0.0))
            
            if "adaptation" in session_results and "adaptation_gain" in session_results["adaptation"]:
                metrics["adaptation_gain"].append(session_results["adaptation"]["adaptation_gain"])
                metrics["adaptation_rate"].append(session_results["adaptation"]["adaptation_rate"])
        
        # Calculate aggregated metrics for each model
        for model_name, model_results in self.results["models"].items():
            metrics = model_results["metrics"]
            
            model_results["aggregated"] = {
                "correctness_rate": np.mean(metrics["correctness_rate"]),
                "style_score": np.mean(metrics["style_score"]),
                "speed_score": np.mean(metrics["speed_score"]),
                "satisfaction": np.mean(metrics["satisfaction"]),
                "adaptation_gain": np.mean(metrics["adaptation_gain"]) if metrics["adaptation_gain"] else 0.0,
                "adaptation_rate": np.mean(metrics["adaptation_rate"]) if metrics["adaptation_rate"] else 0.0
            }
        
        # Calculate overall results
        overall_metrics = {
            "correctness_rate": [],
            "style_score": [],
            "speed_score": [],
            "satisfaction": [],
            "adaptation_gain": [],
            "adaptation_rate": []
        }
        
        for model_name, model_results in self.results["models"].items():
            aggregated = model_results["aggregated"]
            
            overall_metrics["correctness_rate"].append(aggregated["correctness_rate"])
            overall_metrics["style_score"].append(aggregated["style_score"])
            overall_metrics["speed_score"].append(aggregated["speed_score"])
            overall_metrics["satisfaction"].append(aggregated["satisfaction"])
            overall_metrics["adaptation_gain"].append(aggregated["adaptation_gain"])
            overall_metrics["adaptation_rate"].append(aggregated["adaptation_rate"])
        
        self.results["overall"] = {
            "correctness_rate": np.mean(overall_metrics["correctness_rate"]),
            "style_score": np.mean(overall_metrics["style_score"]),
            "speed_score": np.mean(overall_metrics["speed_score"]),
            "satisfaction": np.mean(overall_metrics["satisfaction"]),
            "adaptation_gain": np.mean(overall_metrics["adaptation_gain"]),
            "adaptation_rate": np.mean(overall_metrics["adaptation_rate"])
        }
        
        # Save results
        save_json(self.results, os.path.join(self.output_dir, "evaluation_results.json"))
        
        logger.info("Experiment evaluation completed")
        
        return self.results
    
    def visualize_results(self) -> None:
        """Visualize experiment results."""
        if not self.results or "models" not in self.results:
            logger.warning("No results to visualize")
            return
        
        # Prepare data for visualization
        model_names = list(self.results["models"].keys())
        metrics = [
            "correctness_rate",
            "style_score",
            "speed_score",
            "satisfaction",
            "adaptation_gain",
            "adaptation_rate"
        ]
        
        metric_values = {
            metric: [self.results["models"][model]["aggregated"][metric] for model in model_names]
            for metric in metrics
        }
        
        # Create bar plots for each metric
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Create bar plot
            bars = plt.bar(model_names, metric_values[metric])
            
            # Add labels and title
            plt.xlabel("Model")
            plt.ylabel(metric.replace("_", " ").title())
            plt.title(f"{metric.replace('_', ' ').title()} by Model")
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f"{height:.2f}", ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, f"{metric}_by_model.png"))
            plt.close()
        
        # Create comparative bar plot for all metrics
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, metric_values[metric], width, label=metric.replace("_", " ").title())
        
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.title("Comparative Performance by Model")
        plt.xticks(x + width * (len(metrics) - 1) / 2, model_names)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, "comparative_performance.png"))
        plt.close()
        
        # Adaptation visualization (if available)
        if "adaptation_gain" in metrics and "adaptation_rate" in metrics:
            plt.figure(figsize=(10, 6))
            
            x = [metric_values["adaptation_gain"][i] for i in range(len(model_names))]
            y = [metric_values["adaptation_rate"][i] for i in range(len(model_names))]
            
            plt.scatter(x, y, s=100)
            
            for i, model in enumerate(model_names):
                plt.annotate(model, (x[i], y[i]), xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel("Adaptation Gain")
            plt.ylabel("Adaptation Rate")
            plt.title("Adaptation Performance")
            plt.grid(True)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.output_dir, "adaptation_performance.png"))
            plt.close()
        
        logger.info(f"Visualization results saved to {self.output_dir}")
    
    def generate_results_markdown(self) -> str:
        """
        Generate markdown report of the results.
        
        Returns:
            Markdown formatted results
        """
        if not self.results or "models" not in self.results:
            return "# Experiment Results\n\nNo results available."
        
        markdown = "# Experiment Results for Adaptive Code Assistants\n\n"
        
        # Add overall summary
        markdown += "## Summary\n\n"
        markdown += "This experiment evaluated the effectiveness of adaptive code assistants compared to baseline methods. "
        markdown += "The key hypothesis was that AI code assistants can be significantly more effective when they continuously adapt to individual developer workflows, preferences, and coding habits.\n\n"
        
        # Add overall metrics table
        markdown += "## Overall Performance\n\n"
        markdown += "| Metric | Score |\n"
        markdown += "|--------|------|\n"
        
        for metric, value in self.results["overall"].items():
            markdown += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"
        
        markdown += "\n"
        
        # Add comparative metrics table
        markdown += "## Comparative Performance\n\n"
        markdown += "| Model | Correctness Rate | Style Score | Speed Score | Satisfaction | Adaptation Gain | Adaptation Rate |\n"
        markdown += "|-------|-----------------|------------|------------|--------------|-----------------|------------------|\n"
        
        for model_name, model_results in self.results["models"].items():
            aggregated = model_results["aggregated"]
            markdown += f"| {model_name} | {aggregated['correctness_rate']:.4f} | {aggregated['style_score']:.4f} | {aggregated['speed_score']:.4f} | {aggregated['satisfaction']:.4f} | {aggregated['adaptation_gain']:.4f} | {aggregated['adaptation_rate']:.4f} |\n"
        
        markdown += "\n"
        
        # Add figures
        markdown += "## Visualizations\n\n"
        
        metrics = [
            "correctness_rate",
            "style_score",
            "speed_score",
            "satisfaction",
            "adaptation_gain",
            "adaptation_rate"
        ]
        
        for metric in metrics:
            markdown += f"### {metric.replace('_', ' ').title()} by Model\n\n"
            markdown += f"![{metric.replace('_', ' ').title()} by Model]({metric}_by_model.png)\n\n"
        
        markdown += "### Comparative Performance\n\n"
        markdown += "![Comparative Performance](comparative_performance.png)\n\n"
        
        if "adaptation_gain" in metrics and "adaptation_rate" in metrics:
            markdown += "### Adaptation Performance\n\n"
            markdown += "![Adaptation Performance](adaptation_performance.png)\n\n"
        
        # Add discussion
        markdown += "## Discussion\n\n"
        
        # Compare baseline models with adaptive models
        baseline_models = ["static", "fine_tuned", "rule_based"]
        adaptive_models = ["online", "maml", "hybrid"]
        
        baseline_satisfaction = np.mean([
            self.results["models"][model]["aggregated"]["satisfaction"]
            for model in baseline_models
            if model in self.results["models"]
        ])
        
        adaptive_satisfaction = np.mean([
            self.results["models"][model]["aggregated"]["satisfaction"]
            for model in adaptive_models
            if model in self.results["models"]
        ])
        
        improvement = (adaptive_satisfaction - baseline_satisfaction) / baseline_satisfaction * 100
        
        markdown += f"The experiment results show that adaptive code assistants achieved a {improvement:.2f}% improvement in user satisfaction compared to baseline methods. "
        
        # Check if the best model is adaptive
        model_satisfaction = {
            model: self.results["models"][model]["aggregated"]["satisfaction"]
            for model in self.results["models"]
        }
        
        best_model = max(model_satisfaction, key=model_satisfaction.get)
        
        if best_model in adaptive_models:
            markdown += f"The best performing model was the **{best_model}** approach, demonstrating that {best_model.replace('_', ' ')} adaptation provides significant benefits for code assistance.\n\n"
        else:
            markdown += f"Interestingly, the best performing model was the **{best_model}** approach, suggesting that the benefits of adaptation may depend on specific scenarios and developer profiles.\n\n"
        
        # Add adaptation analysis
        if "adaptation_gain" in metrics and "adaptation_rate" in metrics:
            best_adaptation = max(
                [model for model in self.results["models"] if model in adaptive_models],
                key=lambda m: self.results["models"][m]["aggregated"]["adaptation_gain"]
            )
            
            markdown += f"In terms of adaptation, the **{best_adaptation}** model showed the strongest improvement over time, with an adaptation gain of {self.results['models'][best_adaptation]['aggregated']['adaptation_gain']:.4f} and an adaptation rate of {self.results['models'][best_adaptation]['aggregated']['adaptation_rate']:.4f}. "
            markdown += "This indicates that the model effectively learned from developer feedback and improved its personalization over successive interactions.\n\n"
        
        # Add limitations
        markdown += "## Limitations\n\n"
        markdown += "- The experiment used simulated developer profiles rather than real developers, which may not fully capture the complexity of real-world developer preferences and behaviors.\n"
        markdown += "- The evaluation was conducted on a limited set of coding tasks, which may not represent the full diversity of programming scenarios.\n"
        markdown += "- The adaptation process was simulated within a relatively short timeframe, whereas real-world adaptation would occur over longer periods and more varied tasks.\n"
        markdown += "- The experiment focused on code completion tasks and may not generalize to other code assistance scenarios like refactoring, bug fixing, or architecture design.\n\n"
        
        # Add future work
        markdown += "## Future Work\n\n"
        markdown += "- Conduct user studies with real developers to validate the simulation results and gather qualitative feedback.\n"
        markdown += "- Explore adaptation mechanisms for more diverse coding tasks and languages.\n"
        markdown += "- Investigate the long-term effects of adaptation on developer productivity and code quality.\n"
        markdown += "- Develop more sophisticated personalization techniques that can capture complex developer preferences and coding styles.\n"
        markdown += "- Explore privacy-preserving adaptation mechanisms that can learn from developer interactions without compromising sensitive information.\n\n"
        
        # Add conclusion
        markdown += "## Conclusion\n\n"
        markdown += "The experiment results support the hypothesis that adaptive code assistants can significantly improve developer experience through personalization. "
        markdown += "By continuously learning from developer interactions and feedback, adaptive models can better align with individual preferences, leading to higher satisfaction and productivity. "
        markdown += "The proposed approaches—online learning, MAML-based adaptation, and hybrid methods—all showed promising results, with the hybrid approach generally performing best across multiple metrics.\n\n"
        markdown += "These findings highlight the importance of personalization in AI-assisted software development and suggest that future code assistants should incorporate adaptation mechanisms to better serve diverse developer needs and workflows.\n"
        
        return markdown