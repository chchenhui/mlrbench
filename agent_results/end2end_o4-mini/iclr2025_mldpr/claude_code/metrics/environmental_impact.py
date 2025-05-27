"""
Environmental impact metrics module

This module implements metrics for evaluating the environmental impact of machine learning models,
including energy consumption, carbon emissions, and computational efficiency.
"""

import time
import numpy as np
from typing import Dict, Optional, Any, Union
import os
import platform
import psutil
import warnings


class ResourceMonitor:
    """
    Class to monitor computational resources during model training and inference.
    """
    
    def __init__(self, gpu_available: bool = False):
        """
        Initialize the resource monitor.
        
        Args:
            gpu_available: Whether GPU is available
        """
        self.gpu_available = gpu_available
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None
        self.initial_cpu_percent = None
        
        # Try to import GPU monitoring tools if GPU is available
        if self.gpu_available:
            try:
                import torch
                self.torch_available = True
            except ImportError:
                self.torch_available = False
                warnings.warn("PyTorch not available for GPU monitoring.")
    
    def start_monitoring(self):
        """
        Start monitoring resources.
        """
        self.start_time = time.time()
        
        # CPU memory monitoring
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
        # CPU usage
        self.initial_cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # GPU monitoring if available
        if self.gpu_available and self.torch_available:
            try:
                import torch
                self.start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                self.peak_gpu_memory = self.start_gpu_memory
            except Exception as e:
                warnings.warn(f"Failed to monitor GPU: {str(e)}")
    
    def update_peak_memory(self):
        """
        Update peak memory usage.
        """
        current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.peak_memory, current_memory)
        
        if self.gpu_available and self.torch_available:
            try:
                import torch
                current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                self.peak_gpu_memory = max(self.peak_gpu_memory, current_gpu_memory)
            except Exception:
                pass
    
    def stop_monitoring(self):
        """
        Stop monitoring resources.
        
        Returns:
            dict: Dictionary of resource usage metrics
        """
        self.end_time = time.time()
        self.update_peak_memory()
        
        # Calculate metrics
        elapsed_time = self.end_time - self.start_time
        memory_usage = self.peak_memory - self.start_memory
        
        metrics = {
            'elapsed_time_seconds': float(elapsed_time),
            'cpu_memory_usage_mb': float(memory_usage),
            'peak_memory_mb': float(self.peak_memory)
        }
        
        # Add GPU metrics if available
        if self.gpu_available and self.torch_available:
            try:
                import torch
                gpu_memory_usage = self.peak_gpu_memory - self.start_gpu_memory
                metrics['gpu_memory_usage_mb'] = float(gpu_memory_usage)
                metrics['peak_gpu_memory_mb'] = float(self.peak_gpu_memory)
            except Exception:
                pass
        
        return metrics


def estimate_energy_consumption(
    elapsed_time: float,
    cpu_power_draw: float = 65.0,  # W
    gpu_power_draw: Optional[float] = None,  # W
    gpu_utilization: Optional[float] = None  # 0-1
) -> Dict[str, float]:
    """
    Estimate energy consumption based on elapsed time and approximate power draw.
    
    Args:
        elapsed_time: Time elapsed in seconds
        cpu_power_draw: Estimated CPU power draw in watts (default: 65W)
        gpu_power_draw: Estimated GPU power draw in watts (if GPU is used)
        gpu_utilization: Estimated GPU utilization (0-1)
        
    Returns:
        dict: Dictionary of energy consumption metrics
    """
    # Convert seconds to hours
    hours = elapsed_time / 3600
    
    # Calculate CPU energy consumption
    cpu_energy_kwh = (cpu_power_draw * hours) / 1000  # kWh
    
    metrics = {
        'cpu_energy_kwh': float(cpu_energy_kwh),
        'total_energy_kwh': float(cpu_energy_kwh)
    }
    
    # Add GPU energy consumption if GPU is used
    if gpu_power_draw is not None and gpu_utilization is not None:
        gpu_energy_kwh = (gpu_power_draw * gpu_utilization * hours) / 1000  # kWh
        metrics['gpu_energy_kwh'] = float(gpu_energy_kwh)
        metrics['total_energy_kwh'] += float(gpu_energy_kwh)
    
    return metrics


def estimate_carbon_emissions(
    energy_kwh: float,
    carbon_intensity: float = 475.0  # g CO2e/kWh (global average)
) -> float:
    """
    Estimate carbon emissions based on energy consumption.
    
    Args:
        energy_kwh: Energy consumption in kWh
        carbon_intensity: Carbon intensity in g CO2e/kWh (default: global average)
        
    Returns:
        float: Carbon emissions in kg CO2e
    """
    emissions_g = energy_kwh * carbon_intensity
    emissions_kg = emissions_g / 1000
    return float(emissions_kg)


def calculate_normalized_metrics(
    metrics: Dict[str, float],
    num_samples: int,
    model_size_mb: Optional[float] = None
) -> Dict[str, float]:
    """
    Calculate normalized metrics per sample and per MB of model size.
    
    Args:
        metrics: Dictionary of resource metrics
        num_samples: Number of samples processed
        model_size_mb: Model size in MB (optional)
        
    Returns:
        dict: Dictionary of normalized metrics
    """
    normalized_metrics = {}
    
    # Normalize per sample
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isnan(value):
            normalized_metrics[f"{key}_per_sample"] = float(value / num_samples)
    
    # Normalize per MB of model size if provided
    if model_size_mb is not None and model_size_mb > 0:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                normalized_metrics[f"{key}_per_mb"] = float(value / model_size_mb)
    
    return normalized_metrics


def calculate_environmental_impact(
    monitor_results: Dict[str, float],
    num_samples: int,
    model_size_mb: Optional[float] = None,
    gpu_power_draw: Optional[float] = None,
    gpu_utilization: Optional[float] = None,
    carbon_intensity: float = 475.0
) -> Dict[str, float]:
    """
    Calculate comprehensive environmental impact metrics.
    
    Args:
        monitor_results: Results from resource monitoring
        num_samples: Number of samples processed
        model_size_mb: Model size in MB (optional)
        gpu_power_draw: Estimated GPU power draw in watts (if GPU is used)
        gpu_utilization: Estimated GPU utilization (0-1)
        carbon_intensity: Carbon intensity in g CO2e/kWh
        
    Returns:
        dict: Dictionary of environmental impact metrics
    """
    # Start with the monitor results
    metrics = monitor_results.copy()
    
    # Estimate energy consumption
    energy_metrics = estimate_energy_consumption(
        monitor_results['elapsed_time_seconds'],
        gpu_power_draw=gpu_power_draw,
        gpu_utilization=gpu_utilization
    )
    metrics.update(energy_metrics)
    
    # Estimate carbon emissions
    carbon_emissions = estimate_carbon_emissions(
        metrics['total_energy_kwh'],
        carbon_intensity=carbon_intensity
    )
    metrics['carbon_emissions_kg'] = carbon_emissions
    
    # Calculate normalized metrics
    normalized_metrics = calculate_normalized_metrics(
        metrics, num_samples, model_size_mb
    )
    metrics.update(normalized_metrics)
    
    return metrics