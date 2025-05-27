"""
Models module for Reasoning Uncertainty Networks (RUNs) experiment.

This module implements the core components of the RUNs framework:
1. Reasoning Graph Constructor
2. Uncertainty Initializer
3. Belief Propagation Engine
4. Hallucination Detection Module
"""
import os
import json
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import random
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import anthropic

from config import RUNS_CONFIG, LLM_CONFIG, MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMInterface:
    """Interface for interacting with LLMs (Claude, GPT, etc.)."""
    
    def __init__(self, model_config: Dict = None):
        """
        Initialize the LLM interface.
        
        Args:
            model_config: Configuration for the LLM
        """
        if model_config is None:
            model_config = LLM_CONFIG["primary_model"]
            
        self.model_config = model_config
        self.provider = model_config["provider"]
        self.model_name = model_config["name"]
        
        # Initialize the appropriate client based on provider
        if self.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not found")
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.provider == "openai":
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not found")
            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        logger.info(f"Initialized LLM interface for {self.model_name} from {self.provider}")
    
    def generate(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        if temperature is None:
            temperature = self.model_config.get("temperature", 0.2)
        
        if max_tokens is None:
            max_tokens = self.model_config.get("max_tokens", 1000)
        
        try:
            if self.provider == "anthropic":
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""
    
    def get_confidence(self, assertion: str, context: str = "") -> float:
        """
        Get the LLM's confidence in an assertion.
        
        Args:
            assertion: The assertion to evaluate
            context: Optional context for the assertion
            
        Returns:
            Confidence score between 0 and 1
        """
        prompt = f"""
Based on the following context and assertion, please evaluate the confidence level (from 0 to 100%) 
that the assertion is accurate. Provide ONLY a number between 0 and 100, no other text.

Context:
{context}

Assertion:
{assertion}

Confidence (0-100%):
"""
        
        try:
            response = self.generate(prompt, temperature=0.0)
            # Extract the number from the response
            confidence = float(response.strip().replace('%', '')) / 100
            return min(max(confidence, 0.0), 1.0)  # Ensure value is between 0 and 1
        except:
            logger.warning(f"Failed to extract confidence for assertion: {assertion}")
            return 0.5  # Default to neutral confidence
    
    def generate_variations(self, assertion: str, n: int = 5) -> List[str]:
        """
        Generate variations of an assertion to assess semantic consistency.
        
        Args:
            assertion: The original assertion
            n: Number of variations to generate
            
        Returns:
            List of assertion variations
        """
        prompt = f"""
Please rephrase the following assertion in {n} different ways, maintaining the same meaning
but using different words, sentence structures, or phrasing.

Original assertion:
{assertion}

Generate {n} different rephrased versions, labeled as 1., 2., etc.:
"""
        
        try:
            response = self.generate(prompt)
            # Parse the variations from the response
            variations = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('- ')):
                    # Remove the leading number/bullet and any trailing/leading whitespace
                    clean_line = line.lstrip('0123456789.- \t')
                    if clean_line:
                        variations.append(clean_line)
            
            # If we didn't get enough variations, pad with the original
            while len(variations) < n:
                variations.append(assertion)
            
            # If we got too many, truncate
            variations = variations[:n]
            
            return variations
        except:
            logger.warning(f"Failed to generate variations for assertion: {assertion}")
            return [assertion] * n


class ReasoningGraphConstructor:
    """
    Constructs a reasoning graph from LLM outputs, where nodes are assertions
    and edges represent logical dependencies between assertions.
    """
    
    def __init__(self, llm_interface: LLMInterface = None):
        """
        Initialize the reasoning graph constructor.
        
        Args:
            llm_interface: Interface for LLM interactions
        """
        self.config = RUNS_CONFIG["reasoning_graph"]
        self.llm = llm_interface or LLMInterface()
        self.graph = None
    
    def extract_reasoning_steps(self, question: str, context: str = "") -> List[str]:
        """
        Extract reasoning steps from LLM for a given question.
        
        Args:
            question: The question to reason about
            context: Additional context for the question
            
        Returns:
            List of reasoning steps as strings
        """
        prompt_template = self.config["prompt_template"]
        
        # Replace placeholders in the template
        prompt = prompt_template.replace("[problem]", f"{question}\n\nContext: {context}")
        
        response = self.llm.generate(prompt)
        
        # Parse the response to extract reasoning steps
        steps = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line.lower().startswith("step") and ":" in line:
                # Extract the step content after the colon
                step_content = line.split(":", 1)[1].strip()
                if step_content:
                    steps.append(step_content)
            elif line.lower().startswith("conclusion") and ":" in line:
                # Include the conclusion as the final step
                conclusion = line.split(":", 1)[1].strip()
                if conclusion:
                    steps.append(conclusion)
        
        return steps
    
    def identify_dependencies(self, steps: List[str]) -> List[List[int]]:
        """
        Identify dependencies between reasoning steps.
        
        Args:
            steps: List of reasoning steps
            
        Returns:
            List of lists, where dependencies[i] contains indices of steps that step i depends on
        """
        dependencies = [[] for _ in range(len(steps))]
        
        # First step has no dependencies
        if len(steps) <= 1:
            return dependencies
        
        # For remaining steps, ask the LLM to identify dependencies
        for i in range(1, len(steps)):
            prompt = f"""
I'm analyzing a chain of reasoning with the following steps:

{self._format_steps_for_prompt(steps)}

For step {i+1}: "{steps[i]}"

Which previous steps (ONLY specify step numbers, e.g., 1, 2) does this step DIRECTLY depend on?
Your answer should ONLY include step numbers separated by commas. No other text.
"""
            response = self.llm.generate(prompt, temperature=0.0)
            
            # Parse dependency indices from response
            try:
                # Extract digits from response
                step_indices = [int(s) - 1 for s in response.strip().replace(',', ' ').split() 
                                if s.isdigit() and 0 < int(s) <= i]
                
                # Add valid dependencies
                for idx in step_indices:
                    if 0 <= idx < i:  # Ensure we only have dependencies on previous steps
                        dependencies[i].append(idx)
            except:
                logger.warning(f"Failed to parse dependencies for step {i+1}")
                # Default to depending on the previous step
                dependencies[i].append(i-1)
        
        return dependencies
    
    def _format_steps_for_prompt(self, steps: List[str]) -> str:
        """Format reasoning steps for inclusion in prompts."""
        return "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
    
    def construct_graph(self, question: str, context: str = "") -> nx.DiGraph:
        """
        Construct a reasoning graph for a given question.
        
        Args:
            question: The question to reason about
            context: Additional context for the question
            
        Returns:
            NetworkX directed graph representing the reasoning
        """
        # Extract reasoning steps
        steps = self.extract_reasoning_steps(question, context)
        logger.info(f"Extracted {len(steps)} reasoning steps")
        
        if not steps:
            # Return empty graph if no steps were extracted
            self.graph = nx.DiGraph()
            return self.graph
        
        # Identify dependencies between steps
        dependencies = self.identify_dependencies(steps)
        
        # Construct the graph
        G = nx.DiGraph()
        
        # Add nodes (reasoning steps)
        for i, step in enumerate(steps):
            G.add_node(i, assertion=step, step_num=i+1)
        
        # Add edges (dependencies)
        for i, deps in enumerate(dependencies):
            for dep in deps:
                G.add_edge(dep, i)
        
        self.graph = G
        return G
    
    def visualize_graph(self, output_path: Optional[str] = None) -> None:
        """
        Visualize the reasoning graph.
        
        Args:
            output_path: Optional path to save the visualization
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            logger.warning("No graph to visualize")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create position layout
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color="lightblue")
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, arrowsize=20, width=2, edge_color="gray")
        
        # Draw labels
        labels = {i: f"Step {i+1}" for i in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10)
        
        plt.title("Reasoning Graph")
        plt.axis("off")
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved reasoning graph visualization to {output_path}")
        
        plt.close()


class UncertaintyInitializer:
    """
    Initializes uncertainty distributions for each node in the reasoning graph
    based on LLM confidence, semantic similarity, and knowledge verification.
    """
    
    def __init__(self, llm_interface: LLMInterface = None, embedding_model: str = None):
        """
        Initialize the uncertainty initializer.
        
        Args:
            llm_interface: Interface for LLM interactions
            embedding_model: Name of the embedding model for semantic similarity
        """
        self.config = RUNS_CONFIG["uncertainty_initializer"]
        self.llm = llm_interface or LLMInterface()
        
        # Initialize embedding model for semantic similarity
        if embedding_model is None:
            embedding_model = LLM_CONFIG["embedding_model"]["name"]
        
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"Initialized embedding model: {embedding_model}")
    
    def compute_semantic_similarity(self, assertion: str) -> float:
        """
        Compute semantic similarity score for an assertion by generating variations
        and comparing their embeddings.
        
        Args:
            assertion: The assertion to evaluate
            
        Returns:
            Similarity score between 0 and 1
        """
        # Generate variations of the assertion
        variations = self.llm.generate_variations(assertion, n=self.config["num_variations"])
        
        # Compute embeddings for all variations
        embeddings = self.embedding_model.encode(variations)
        
        # Compute pairwise cosine similarities
        similarities = []
        n = len(variations)
        for i in range(n):
            for j in range(i+1, n):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(similarity)
        
        # Return average similarity
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return float(avg_similarity)
        else:
            return 0.5  # Default if no similarities could be computed
    
    def verify_assertion(self, assertion: str, context: str) -> float:
        """
        Verify an assertion against provided context.
        In a full implementation, this would use retrieval-augmented verification.
        
        Args:
            assertion: The assertion to verify
            context: Context to verify against
            
        Returns:
            Verification score between 0 and 1
        """
        # For simplicity, we'll use the LLM to verify
        # In a real implementation, we would use a more sophisticated approach
        
        prompt = f"""
Based on ONLY the following context, assess whether the assertion is verifiable and true.
Rate on a scale from 0 to 100, where:
- 0 means the assertion is completely contradicted by or unverifiable from the context
- 100 means the assertion is fully supported by the context
Provide ONLY a number between 0 and 100, no other text.

Context:
{context}

Assertion:
{assertion}

Verification score (0-100):
"""
        
        try:
            response = self.llm.generate(prompt, temperature=0.0)
            # Extract the number from the response
            verification = float(response.strip().replace('%', '')) / 100
            return min(max(verification, 0.0), 1.0)  # Ensure value is between 0 and 1
        except:
            logger.warning(f"Failed to verify assertion: {assertion}")
            return 0.5  # Default to neutral verification
    
    def initialize_uncertainties(self, graph: nx.DiGraph, context: str = "") -> nx.DiGraph:
        """
        Initialize uncertainty distributions for each node in the graph.
        
        Args:
            graph: Reasoning graph
            context: Context for verification
            
        Returns:
            Graph with initialized uncertainty distributions
        """
        # Make a copy of the graph to avoid modifying the original
        graph = graph.copy()
        
        for node_id in tqdm(graph.nodes(), desc="Initializing uncertainties"):
            assertion = graph.nodes[node_id]["assertion"]
            
            # Get LLM confidence
            if self.config["use_llm_confidence"]:
                confidence = self.llm.get_confidence(assertion, context)
            else:
                confidence = 0.5  # Default confidence
            
            # Compute semantic similarity
            if self.config["use_semantic_similarity"]:
                similarity = self.compute_semantic_similarity(assertion)
            else:
                similarity = 0.5  # Default similarity
            
            # Verify against context
            if self.config["use_knowledge_verification"] and context:
                verification = self.verify_assertion(assertion, context)
            else:
                verification = 0.5  # Default verification
            
            # Compute concentration parameter
            # Higher consistency -> higher concentration -> lower variance
            consistency = (similarity + verification) / 2
            concentration = 2 + 20 * (consistency - 0.5)  # Scale to reasonable range [2, 12]
            concentration = max(2, concentration)  # Ensure minimum concentration
            
            # Set the Beta distribution parameters
            alpha = confidence * concentration
            beta = (1 - confidence) * concentration
            
            # Store the uncertainty parameters in the node
            graph.nodes[node_id].update({
                "confidence": confidence,
                "similarity": similarity,
                "verification": verification,
                "concentration": concentration,
                "alpha": alpha,
                "beta": beta,
                "mean": confidence,
                "variance": (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
            })
        
        return graph


class BeliefPropagationEngine:
    """
    Updates uncertainty values across the graph using belief propagation
    to propagate uncertainty from premises to conclusions.
    """
    
    def __init__(self):
        """Initialize the belief propagation engine."""
        self.config = RUNS_CONFIG["belief_propagation"]
        self.iterations = 0
    
    def propagate_beliefs(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Propagate uncertainty through the reasoning graph.
        
        Args:
            graph: Reasoning graph with initialized uncertainties
            
        Returns:
            Graph with updated uncertainty distributions
        """
        # Make a copy of the graph to avoid modifying the original
        graph = graph.copy()
        
        # Store initial uncertainty parameters
        for node_id in graph.nodes():
            graph.nodes[node_id]["alpha_0"] = graph.nodes[node_id]["alpha"]
            graph.nodes[node_id]["beta_0"] = graph.nodes[node_id]["beta"]
        
        # Iterate until convergence or max iterations
        for iteration in range(self.config["max_iterations"]):
            self.iterations = iteration + 1
            
            # Copy current parameters for comparison
            old_params = {
                node_id: (graph.nodes[node_id]["alpha"], graph.nodes[node_id]["beta"])
                for node_id in graph.nodes()
            }
            
            # Update each node based on its parents
            for node_id in graph.nodes():
                parents = list(graph.predecessors(node_id))
                
                if not parents:
                    continue  # Skip nodes without parents
                
                # Initialize with prior parameters
                alpha_new = graph.nodes[node_id]["alpha_0"]
                beta_new = graph.nodes[node_id]["beta_0"]
                
                # Collect messages from parents
                for parent_id in parents:
                    # Get edge weight (default if not specified)
                    edge_weight = graph.edges.get((parent_id, node_id), {}).get(
                        "weight", self.config["edge_weight_default"]
                    )
                    
                    # Get parent parameters
                    alpha_parent = graph.nodes[parent_id]["alpha"]
                    beta_parent = graph.nodes[parent_id]["beta"]
                    
                    # Update parameters based on parent message
                    # Use a weighted contribution based on the edge weight
                    alpha_new += edge_weight * (alpha_parent - 1) / len(parents)
                    beta_new += edge_weight * (beta_parent - 1) / len(parents)
                
                # Ensure parameters remain valid
                alpha_new = max(0.01, alpha_new)
                beta_new = max(0.01, beta_new)
                
                # Update node parameters
                graph.nodes[node_id].update({
                    "alpha": alpha_new,
                    "beta": beta_new,
                    "mean": alpha_new / (alpha_new + beta_new),
                    "variance": (alpha_new * beta_new) / ((alpha_new + beta_new)**2 * (alpha_new + beta_new + 1))
                })
            
            # Check for convergence
            max_diff = 0
            for node_id in graph.nodes():
                alpha_old, beta_old = old_params[node_id]
                alpha_new = graph.nodes[node_id]["alpha"]
                beta_new = graph.nodes[node_id]["beta"]
                
                # Compute parameter difference
                diff = abs(alpha_new - alpha_old) + abs(beta_new - beta_old)
                max_diff = max(max_diff, diff)
            
            if max_diff < self.config["convergence_threshold"]:
                logger.info(f"Belief propagation converged after {iteration + 1} iterations")
                break
        
        # Calculate changes in uncertainty
        for node_id in graph.nodes():
            mean_0 = graph.nodes[node_id]["alpha_0"] / (graph.nodes[node_id]["alpha_0"] + graph.nodes[node_id]["beta_0"])
            var_0 = (graph.nodes[node_id]["alpha_0"] * graph.nodes[node_id]["beta_0"]) / (
                (graph.nodes[node_id]["alpha_0"] + graph.nodes[node_id]["beta_0"])**2 * 
                (graph.nodes[node_id]["alpha_0"] + graph.nodes[node_id]["beta_0"] + 1)
            )
            
            graph.nodes[node_id].update({
                "mean_change": graph.nodes[node_id]["mean"] - mean_0,
                "variance_change": graph.nodes[node_id]["variance"] - var_0
            })
        
        return graph
    
    def visualize_propagation(self, graph: nx.DiGraph, output_path: Optional[str] = None) -> None:
        """
        Visualize the uncertainty propagation in the graph.
        
        Args:
            graph: Reasoning graph after belief propagation
            output_path: Optional path to save the visualization
        """
        if graph is None or graph.number_of_nodes() == 0:
            logger.warning("No graph to visualize")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create position layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Calculate node color based on confidence
        confidence_values = [graph.nodes[n]["mean"] for n in graph.nodes()]
        cmap = plt.cm.RdYlGn  # Red (low confidence) to Green (high confidence)
        
        # Calculate node size based on uncertainty variance
        variance_values = [1000 * (0.5 + graph.nodes[n]["variance"]) for n in graph.nodes()]
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            graph, pos, 
            node_size=variance_values,
            node_color=confidence_values,
            cmap=cmap,
            vmin=0, vmax=1
        )
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, arrowsize=20, width=2, edge_color="gray")
        
        # Draw labels
        labels = {i: f"Step {i+1}\n{graph.nodes[i]['mean']:.2f}" for i in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)
        
        # Add a colorbar
        plt.colorbar(nodes, label="Confidence")
        
        plt.title(f"Uncertainty Propagation (Iterations: {self.iterations})")
        plt.axis("off")
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved propagation visualization to {output_path}")
        
        plt.close()


class HallucinationDetector:
    """
    Identifies potential hallucinations in the reasoning graph based on
    uncertainty distributions and logical consistency.
    """
    
    def __init__(self):
        """Initialize the hallucination detector."""
        self.config = RUNS_CONFIG["hallucination_detection"]
    
    def check_logical_consistency(self, graph: nx.DiGraph) -> Dict[int, float]:
        """
        Check for logical inconsistencies between assertions in the graph.
        
        Args:
            graph: Reasoning graph
            
        Returns:
            Dictionary mapping node IDs to inconsistency scores
        """
        inconsistency_scores = {node_id: 0.0 for node_id in graph.nodes()}
        
        # In a full implementation, we would use more sophisticated methods
        # For simplicity, we'll consider nodes with high variance change as potentially inconsistent
        for node_id in graph.nodes():
            # Higher variance change might indicate logical inconsistency
            var_change = graph.nodes[node_id].get("variance_change", 0)
            inconsistency_scores[node_id] = min(1.0, max(0.0, var_change * 5))
        
        return inconsistency_scores
    
    def detect_hallucinations(self, graph: nx.DiGraph) -> Tuple[nx.DiGraph, List[int]]:
        """
        Detect potential hallucinations in the reasoning graph.
        
        Args:
            graph: Reasoning graph after belief propagation
            
        Returns:
            Tuple of (updated graph, list of hallucination node IDs)
        """
        # Make a copy of the graph to avoid modifying the original
        graph = graph.copy()
        
        # Check logical consistency
        inconsistency_scores = self.check_logical_consistency(graph)
        
        # Identify potential hallucinations
        hallucination_nodes = []
        
        for node_id in graph.nodes():
            # Get relevant parameters
            mean_confidence = graph.nodes[node_id]["mean"]
            var_change = graph.nodes[node_id].get("variance_change", 0)
            inconsistency = inconsistency_scores[node_id]
            
            # Compute hallucination score
            # H = (1 - mean) * (1 + gamma * var_change) * (1 + delta * inconsistency)
            h_score = (1 - mean_confidence) * \
                      (1 + self.config["gamma"] * abs(var_change)) * \
                      (1 + self.config["delta"] * inconsistency)
            
            # Store hallucination score in the node
            graph.nodes[node_id]["hallucination_score"] = h_score
            
            # Check thresholds for flagging hallucinations
            is_hallucination = False
            
            # Criterion 1: Low confidence
            if mean_confidence < self.config["confidence_threshold"]:
                is_hallucination = True
                graph.nodes[node_id]["hallucination_reason"] = "low_confidence"
            
            # Criterion 2: Significant uncertainty increase
            elif var_change > self.config["uncertainty_increase_threshold"]:
                is_hallucination = True
                graph.nodes[node_id]["hallucination_reason"] = "uncertainty_increase"
            
            # Criterion 3: High logical inconsistency
            elif inconsistency > 0.5:  # Arbitrary threshold for this example
                is_hallucination = True
                graph.nodes[node_id]["hallucination_reason"] = "logical_inconsistency"
            
            # Mark the node and add to list if it's a hallucination
            graph.nodes[node_id]["is_hallucination"] = is_hallucination
            if is_hallucination:
                hallucination_nodes.append(node_id)
        
        return graph, hallucination_nodes
    
    def visualize_hallucinations(self, graph: nx.DiGraph, output_path: Optional[str] = None) -> None:
        """
        Visualize hallucinations in the reasoning graph.
        
        Args:
            graph: Reasoning graph with hallucination detection
            output_path: Optional path to save the visualization
        """
        if graph is None or graph.number_of_nodes() == 0:
            logger.warning("No graph to visualize")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Create position layout
        pos = nx.spring_layout(graph, seed=42)
        
        # Separate hallucination and non-hallucination nodes
        hall_nodes = [n for n in graph.nodes() if graph.nodes[n].get("is_hallucination", False)]
        normal_nodes = [n for n in graph.nodes() if not graph.nodes[n].get("is_hallucination", False)]
        
        # Draw normal nodes
        nx.draw_networkx_nodes(
            graph, pos, 
            nodelist=normal_nodes,
            node_size=700,
            node_color="lightblue"
        )
        
        # Draw hallucination nodes
        nx.draw_networkx_nodes(
            graph, pos, 
            nodelist=hall_nodes,
            node_size=900,
            node_color="red"
        )
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, arrowsize=20, width=2, edge_color="gray")
        
        # Draw labels with hallucination scores for hallucination nodes
        labels = {}
        for i in normal_nodes:
            labels[i] = f"Step {i+1}"
        for i in hall_nodes:
            score = graph.nodes[i].get("hallucination_score", 0)
            reason = graph.nodes[i].get("hallucination_reason", "unknown")
            labels[i] = f"Step {i+1}\nScore: {score:.2f}\n({reason})"
        
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)
        
        plt.title("Hallucination Detection")
        plt.axis("off")
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved hallucination visualization to {output_path}")
        
        plt.close()


class ReasoningUncertaintyNetwork:
    """
    Main class for the Reasoning Uncertainty Networks (RUNs) framework.
    """
    
    def __init__(self, llm_interface: LLMInterface = None):
        """
        Initialize the RUNs framework.
        
        Args:
            llm_interface: Interface for LLM interactions
        """
        self.llm = llm_interface or LLMInterface()
        
        # Initialize components
        self.graph_constructor = ReasoningGraphConstructor(self.llm)
        self.uncertainty_initializer = UncertaintyInitializer(self.llm)
        self.belief_propagation = BeliefPropagationEngine()
        self.hallucination_detector = HallucinationDetector()
        
        self.graph = None
        self.hallucination_nodes = []
    
    def process(self, question: str, context: str = "", visualize: bool = False, output_dir: Optional[str] = None) -> Tuple[nx.DiGraph, List[int]]:
        """
        Process a question through the RUNs framework.
        
        Args:
            question: The question to reason about
            context: Additional context for the question
            visualize: Whether to generate visualizations
            output_dir: Directory to save visualizations
            
        Returns:
            Tuple of (final graph, list of hallucination node IDs)
        """
        # Step 1: Construct the reasoning graph
        start_time = time.time()
        self.graph = self.graph_constructor.construct_graph(question, context)
        logger.info(f"Graph construction took {time.time() - start_time:.2f} seconds")
        
        if visualize and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.graph_constructor.visualize_graph(
                os.path.join(output_dir, "reasoning_graph.png")
            )
        
        # Step 2: Initialize uncertainties
        start_time = time.time()
        self.graph = self.uncertainty_initializer.initialize_uncertainties(self.graph, context)
        logger.info(f"Uncertainty initialization took {time.time() - start_time:.2f} seconds")
        
        # Step 3: Propagate beliefs
        start_time = time.time()
        self.graph = self.belief_propagation.propagate_beliefs(self.graph)
        logger.info(f"Belief propagation took {time.time() - start_time:.2f} seconds")
        
        if visualize and output_dir:
            self.belief_propagation.visualize_propagation(
                self.graph,
                os.path.join(output_dir, "uncertainty_propagation.png")
            )
        
        # Step 4: Detect hallucinations
        start_time = time.time()
        self.graph, self.hallucination_nodes = self.hallucination_detector.detect_hallucinations(self.graph)
        logger.info(f"Hallucination detection took {time.time() - start_time:.2f} seconds")
        
        if visualize and output_dir:
            self.hallucination_detector.visualize_hallucinations(
                self.graph,
                os.path.join(output_dir, "hallucination_detection.png")
            )
        
        return self.graph, self.hallucination_nodes
    
    def get_hallucination_details(self) -> List[Dict]:
        """
        Get details of detected hallucinations.
        
        Returns:
            List of dictionaries with hallucination details
        """
        if self.graph is None:
            return []
        
        hallucination_details = []
        
        for node_id in self.hallucination_nodes:
            node_data = self.graph.nodes[node_id]
            
            hallucination_details.append({
                "step_num": node_data.get("step_num", node_id + 1),
                "assertion": node_data.get("assertion", ""),
                "confidence": node_data.get("mean", 0),
                "initial_confidence": node_data.get("alpha_0", 0) / (node_data.get("alpha_0", 0) + node_data.get("beta_0", 1)),
                "hallucination_score": node_data.get("hallucination_score", 0),
                "hallucination_reason": node_data.get("hallucination_reason", "unknown"),
                "dependencies": list(self.graph.predecessors(node_id))
            })
        
        return hallucination_details
    
    def generate_explanation(self) -> str:
        """
        Generate a human-readable explanation of the hallucination detection.
        
        Returns:
            Explanation string
        """
        if self.graph is None or not self.hallucination_nodes:
            return "No hallucinations detected in the reasoning."
        
        explanation = "Potential hallucinations detected in the reasoning:\n\n"
        
        for node_id in sorted(self.hallucination_nodes):
            node_data = self.graph.nodes[node_id]
            step_num = node_data.get("step_num", node_id + 1)
            assertion = node_data.get("assertion", "")
            score = node_data.get("hallucination_score", 0)
            reason = node_data.get("hallucination_reason", "unknown")
            
            explanation += f"Step {step_num}: \"{assertion}\"\n"
            explanation += f"  - Hallucination score: {score:.2f}\n"
            
            if reason == "low_confidence":
                explanation += "  - Reason: Low confidence in this assertion\n"
            elif reason == "uncertainty_increase":
                explanation += "  - Reason: Uncertainty significantly increased during reasoning\n"
            elif reason == "logical_inconsistency":
                explanation += "  - Reason: Potential logical inconsistency with other assertions\n"
            
            # Include information about dependencies
            dependencies = list(self.graph.predecessors(node_id))
            if dependencies:
                dependency_assertions = [
                    f"Step {self.graph.nodes[dep].get('step_num', dep + 1)}" 
                    for dep in dependencies
                ]
                explanation += f"  - Depends on: {', '.join(dependency_assertions)}\n"
            
            explanation += "\n"
        
        return explanation
    
    def save_results(self, output_path: str) -> None:
        """
        Save the results of the analysis to a JSON file.
        
        Args:
            output_path: Path to save the results
        """
        if self.graph is None:
            logger.warning("No graph to save")
            return
        
        # Convert graph to serializable format
        nodes_data = []
        for node_id in self.graph.nodes():
            node_data = dict(self.graph.nodes[node_id])
            node_data["id"] = node_id
            node_data["dependencies"] = list(self.graph.predecessors(node_id))
            nodes_data.append(node_data)
        
        # Create results dictionary
        results = {
            "nodes": nodes_data,
            "hallucination_nodes": self.hallucination_nodes,
            "hallucination_details": self.get_hallucination_details(),
            "explanation": self.generate_explanation()
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")


# Example usage
if __name__ == "__main__":
    # Test the model components
    print("Testing RUNs components...")
    
    # Initialize LLM interface
    llm = LLMInterface()
    
    # Test question
    question = "What happens when you mix baking soda and vinegar?"
    context = "Baking soda is sodium bicarbonate (NaHCO3). Vinegar contains acetic acid (CH3COOH)."
    
    # Create RUNs instance
    runs = ReasoningUncertaintyNetwork(llm)
    
    # Process the question
    graph, hallucinations = runs.process(
        question, 
        context,
        visualize=True,
        output_dir=MODELS_DIR
    )
    
    print(f"\nDetected {len(hallucinations)} potential hallucinations")
    print(runs.generate_explanation())
    
    # Save results
    runs.save_results(os.path.join(MODELS_DIR, "test_results.json"))
    
    print("RUNs component test complete.")