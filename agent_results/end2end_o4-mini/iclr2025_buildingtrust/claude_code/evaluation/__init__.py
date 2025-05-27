"""
Evaluation module for Cluster-Driven Certified Unlearning.
"""

from .metrics import (
    compute_perplexity,
    compute_knowledge_forgetting_rate,
    compute_knowledge_retention_rate,
    evaluate_downstream_task,
    compute_computational_cost,
    evaluate_membership_inference
)

__all__ = [
    'compute_perplexity',
    'compute_knowledge_forgetting_rate',
    'compute_knowledge_retention_rate',
    'evaluate_downstream_task',
    'compute_computational_cost',
    'evaluate_membership_inference'
]