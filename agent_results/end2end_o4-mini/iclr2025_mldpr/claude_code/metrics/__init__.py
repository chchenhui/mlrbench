"""
Multi-Metric Evaluation Suite (MES)

This package implements a comprehensive suite of metrics for evaluating machine learning models
across multiple dimensions: performance, fairness, robustness, environmental impact, and interpretability.
"""

from .performance import calculate_performance_metrics
from .fairness import calculate_fairness_metrics
from .robustness import calculate_robustness_metrics
from .environmental_impact import (
    ResourceMonitor, 
    calculate_environmental_impact
)
from .interpretability import calculate_interpretability_metrics