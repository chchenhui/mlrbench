#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for UDRA experiments.

This module contains functions for visualizing experiment results,
including learning curves, alignment errors, and uncertainty estimates.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves(baseline_rewards, udra_rewards, window_size=10, 
                         title="Learning Curves", save_path=None):
    """
    Plot learning curves for baseline and UDRA agents.
    
    Args:
        baseline_rewards (list): Episode rewards for baseline agent
        udra_rewards (list): Episode rewards for UDRA agent
        window_size (int): Window size for smoothing
        title (str): Plot title
        save_path (str): Path to save figure, or None to display
    """
    plt.figure(figsize=(10, 6))
    
    # Apply smoothing
    def smooth(y, window):
        box = np.ones(window) / window
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    # Plot baseline rewards
    x = np.arange(len(baseline_rewards))
    y_smooth = smooth(baseline_rewards, window_size)
    plt.plot(x, y_smooth, label="Baseline (RLHF)", color='blue')
    
    # Plot UDRA rewards
    y_smooth = smooth(udra_rewards, window_size)
    plt.plot(x, y_smooth, label="UDRA", color='red')
    
    # Add raw data points with lower opacity
    plt.scatter(x[::20], baseline_rewards[::20], color='blue', alpha=0.3, s=10)
    plt.scatter(x[::20], udra_rewards[::20], color='red', alpha=0.3, s=10)
    
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_alignment_errors(baseline_errors, udra_errors, window_size=10,
                          title="Alignment Errors", save_path=None):
    """
    Plot alignment errors for baseline and UDRA agents.
    
    Args:
        baseline_errors (list): Alignment errors for baseline agent
        udra_errors (list): Alignment errors for UDRA agent
        window_size (int): Window size for smoothing
        title (str): Plot title
        save_path (str): Path to save figure, or None to display
    """
    plt.figure(figsize=(10, 6))
    
    # Apply smoothing
    def smooth(y, window):
        box = np.ones(window) / window
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    # Plot baseline errors
    x = np.arange(len(baseline_errors))
    y_smooth = smooth(baseline_errors, window_size)
    plt.plot(x, y_smooth, label="Baseline (RLHF)", color='blue')
    
    # Plot UDRA errors
    y_smooth = smooth(udra_errors, window_size)
    plt.plot(x, y_smooth, label="UDRA", color='red')
    
    # Add raw data points with lower opacity
    plt.scatter(x[::20], baseline_errors[::20], color='blue', alpha=0.3, s=10)
    plt.scatter(x[::20], udra_errors[::20], color='red', alpha=0.3, s=10)
    
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Alignment Error")
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_uncertainty_calibration(uncertainties, corrections, bins=10,
                                 title="Uncertainty Calibration", save_path=None):
    """
    Plot uncertainty calibration curve.
    
    Args:
        uncertainties (list): Uncertainty estimates from agent
        corrections (list): Binary indicators of corrections (1 if corrected, 0 if not)
        bins (int): Number of bins for grouping uncertainties
        title (str): Plot title
        save_path (str): Path to save figure, or None to display
    """
    plt.figure(figsize=(10, 6))
    
    # Create bins of uncertainty values
    uncertainty_bins = np.linspace(min(uncertainties), max(uncertainties), bins+1)
    bin_indices = np.digitize(uncertainties, uncertainty_bins)
    
    # Compute correction rate for each bin
    bin_correction_rates = []
    bin_centers = []
    
    for bin_idx in range(1, bins+1):
        mask = (bin_indices == bin_idx)
        if sum(mask) > 0:
            correction_rate = np.mean(np.array(corrections)[mask])
            bin_correction_rates.append(correction_rate)
            bin_centers.append((uncertainty_bins[bin_idx-1] + uncertainty_bins[bin_idx]) / 2)
    
    # Plot calibration curve
    plt.scatter(bin_centers, bin_correction_rates, s=100, color='blue')
    plt.plot(bin_centers, bin_correction_rates, color='blue')
    
    # Plot ideal calibration line
    ideal_x = [min(bin_centers), max(bin_centers)]
    ideal_y = [min(bin_centers), max(bin_centers)]
    plt.plot(ideal_x, ideal_y, '--', color='gray', alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Predicted Uncertainty")
    plt.ylabel("Observed Correction Rate")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_preference_evolution(preference_history, true_preference, 
                              title="Preference Weight Evolution", save_path=None):
    """
    Plot the evolution of estimated preference weights over time.
    
    Args:
        preference_history (list): History of estimated preference vectors
        true_preference (np.ndarray): True preference vector
        title (str): Plot title
        save_path (str): Path to save figure, or None to display
    """
    plt.figure(figsize=(12, 6))
    
    # Convert history to array
    preference_array = np.array(preference_history)
    n_weights = preference_array.shape[1]
    
    # Plot each preference weight over time
    for i in range(n_weights):
        plt.plot(preference_array[:, i], label=f"Weight {i+1}")
    
    # Highlight true preference weights
    for i, val in enumerate(true_preference):
        plt.axhline(y=val, color=f'C{i}', linestyle='--', alpha=0.5)
    
    plt.title(title)
    plt.xlabel("Update Step")
    plt.ylabel("Preference Weight")
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_correction_frequency(baseline_corrections, udra_corrections, window_size=100,
                              title="Human Correction Frequency", save_path=None):
    """
    Plot the frequency of human corrections over time.
    
    Args:
        baseline_corrections (list): Corrections for baseline agent
        udra_corrections (list): Corrections for UDRA agent
        window_size (int): Window size for computing frequency
        title (str): Plot title
        save_path (str): Path to save figure, or None to display
    """
    plt.figure(figsize=(10, 6))
    
    # Convert None/not-None to binary
    baseline_binary = [1 if x is not None else 0 for x in baseline_corrections]
    udra_binary = [1 if x is not None else 0 for x in udra_corrections]
    
    # Compute correction frequency in sliding windows
    def compute_frequency(corrections, window):
        freq = []
        for i in range(0, len(corrections), window):
            if i + window <= len(corrections):
                freq.append(sum(corrections[i:i+window]) / window)
        return freq
    
    # Compute frequencies
    baseline_freq = compute_frequency(baseline_binary, window_size)
    udra_freq = compute_frequency(udra_binary, window_size)
    
    # Plot frequencies
    x = np.arange(len(baseline_freq))
    plt.plot(x, baseline_freq, label="Baseline (RLHF)", color='blue', marker='o')
    plt.plot(x, udra_freq, label="UDRA", color='red', marker='o')
    
    plt.title(title)
    plt.xlabel(f"Time Window (each window = {window_size} steps)")
    plt.ylabel("Correction Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_histograms(baseline_values, udra_values, x_label,
                    title="Value Distribution", save_path=None):
    """
    Plot histograms of values for baseline and UDRA.
    
    Args:
        baseline_values (list): Values for baseline agent
        udra_values (list): Values for UDRA agent
        x_label (str): Label for x-axis
        title (str): Plot title
        save_path (str): Path to save figure, or None to display
    """
    plt.figure(figsize=(12, 5))
    
    # Create subplot for baseline
    plt.subplot(1, 2, 1)
    plt.hist(baseline_values, bins=30, alpha=0.7, color='blue')
    plt.title(f"Baseline {title}")
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    
    # Create subplot for UDRA
    plt.subplot(1, 2, 2)
    plt.hist(udra_values, bins=30, alpha=0.7, color='red')
    plt.title(f"UDRA {title}")
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()