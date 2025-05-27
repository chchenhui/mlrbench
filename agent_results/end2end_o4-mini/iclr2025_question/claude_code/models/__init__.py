"""
Model implementations and utilities for SCEC experiments.
"""

from .llm_interface import (
    LLMInterface,
    OpenAILLM,
    AnthropicLLM, 
    HuggingFaceLLM,
    get_llm_interface
)

from .self_consistency import (
    SelfConsistencySampler,
    compute_chain_similarities
)

__all__ = [
    'LLMInterface',
    'OpenAILLM',
    'AnthropicLLM',
    'HuggingFaceLLM',
    'get_llm_interface',
    'SelfConsistencySampler',
    'compute_chain_similarities',
]