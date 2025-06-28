#!/usr/bin/env python3
"""
Script to generate a comprehensive report of experiment results.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from utils import generate_markdown_report, generate_comparison_plots, plot_preference_shifts
from environment import RecommendationEnvironment
from utils import generate_user_preferences

def main():
    """Generate the experiment report from results."""
    # Load configuration
    config_path = "../results/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Load results
    results_path = "../results/experiment_results.json"
    with open(results_path, "r") as f:
        results = json.load(f)
    
    # Generate additional comparison plots
    generate_comparison_plots(results, "../results")
    
    # Generate preference shifts visualization
    # Recreate environment to visualize preference shifts
    user_preferences = generate_user_preferences(
        n_users=config["n_users"],
        n_features=config["n_features"]
    )
    
    env = RecommendationEnvironment(
        n_users=config["n_users"],
        n_items=config["n_items"],
        n_features=config["n_features"],
        user_preferences=user_preferences,
        preference_shift_interval=config["preference_shift_interval"],
        preference_shift_magnitude=config["preference_shift_magnitude"]
    )
    
    # Simulate the same preference shifts as in the experiment
    for i in range(config["n_episodes"] // config["preference_shift_interval"]):
        env.shift_preferences()
    
    # Plot preference shifts
    plot_preference_shifts(env, "../results/preference_shifts.png")
    
    # Generate markdown report
    generate_markdown_report(results, config, "../results/results.md")
    
    # Copy results.md to the main results directory
    with open("../results/results.md", "r") as f:
        content = f.read()
    
    with open("../results.md", "w") as f:
        f.write(content)
    
    print("Report generation completed!")

if __name__ == "__main__":
    main()