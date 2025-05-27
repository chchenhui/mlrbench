#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trust path visualization utilities for Attribution-Guided Training.
Implements methods to visualize the path from generated content to source content.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import re
import os
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

class TrustPathVisualizer:
    """
    Base class for trust path visualization.
    """
    
    def __init__(
        self,
        source_metadata: Optional[Dict[int, Dict[str, str]]] = None,
        output_dir: str = "figures"
    ):
        """
        Initialize the trust path visualizer.
        
        Args:
            source_metadata: Metadata about sources (author, title, etc.)
            output_dir: Directory to save visualizations
        """
        self.source_metadata = source_metadata or {}
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized TrustPathVisualizer, outputs will be saved to {output_dir}")
    
    def _get_source_label(self, source_id: int) -> str:
        """
        Get a label for a source.
        
        Args:
            source_id: Source identifier
            
        Returns:
            Human-readable label for the source
        """
        if source_id in self.source_metadata:
            metadata = self.source_metadata[source_id]
            title = metadata.get("title", f"Source {source_id}")
            author = metadata.get("author", "Unknown")
            return f"{title} (by {author})"
        else:
            return f"Source {source_id}"
    
    def _truncate_text(self, text: str, max_length: int = 50) -> str:
        """
        Truncate text to a maximum length.
        
        Args:
            text: Input text
            max_length: Maximum length in characters
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        else:
            return text[:max_length-3] + "..."
    
    def visualize_trust_path(
        self,
        generated_text: str,
        source_texts: List[str],
        attributions: Dict[str, Any],
        filename: str = "trust_path.png"
    ) -> str:
        """
        Visualize the trust path from generated content to sources.
        
        Args:
            generated_text: Generated text
            source_texts: List of source texts
            attributions: Attribution information
            filename: Output filename
            
        Returns:
            Path to the saved visualization
        """
        raise NotImplementedError("Subclasses must implement visualize_trust_path")

class NetworkTrustPathVisualizer(TrustPathVisualizer):
    """
    Visualizer that shows trust paths as a network graph.
    """
    
    def __init__(
        self,
        source_metadata: Optional[Dict[int, Dict[str, str]]] = None,
        output_dir: str = "figures",
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Initialize the network trust path visualizer.
        
        Args:
            source_metadata: Metadata about sources
            output_dir: Directory to save visualizations
            figsize: Figure size in inches
        """
        super().__init__(source_metadata, output_dir)
        self.figsize = figsize
    
    def visualize_trust_path(
        self,
        generated_text: str,
        source_texts: List[str],
        attributions: Dict[str, Any],
        filename: str = "network_trust_path.png"
    ) -> str:
        """
        Visualize the trust path as a network graph.
        
        Args:
            generated_text: Generated text
            source_texts: List of source texts
            attributions: Attribution information
            filename: Output filename
            
        Returns:
            Path to the saved visualization
        """
        # Create graph
        G = nx.DiGraph()
        
        # Add generated text node
        G.add_node("generated", text=self._truncate_text(generated_text), type="generated")
        
        # Get attributions
        sources = attributions.get("sources", [])
        scores = attributions.get("scores", [])
        
        # Normalize scores if not provided
        if not scores and sources:
            scores = [1.0 / len(sources)] * len(sources)
        
        # Add source nodes and edges
        for i, source_id in enumerate(sources):
            if source_id < len(source_texts):
                source_text = source_texts[source_id]
                source_label = self._get_source_label(source_id)
                
                node_id = f"source_{source_id}"
                G.add_node(node_id, text=self._truncate_text(source_text), 
                          label=source_label, type="source")
                
                # Add edge with weight based on attribution score
                score = scores[i] if i < len(scores) else 0.5
                G.add_edge("generated", node_id, weight=score)
        
        # Draw the graph
        plt.figure(figsize=self.figsize)
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with different colors for generated and source
        generated_nodes = [n for n, attr in G.nodes(data=True) if attr["type"] == "generated"]
        source_nodes = [n for n, attr in G.nodes(data=True) if attr["type"] == "source"]
        
        nx.draw_networkx_nodes(G, pos, nodelist=generated_nodes, 
                              node_color="skyblue", node_size=2000, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, 
                              node_color="lightgreen", node_size=1500, alpha=0.8)
        
        # Draw edges with width based on score
        edges = G.edges(data=True)
        edge_widths = [e[2]["weight"] * 5 for e in edges]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7, 
                              edge_color="gray", arrows=True, arrowsize=20)
        
        # Draw node labels
        labels = {}
        for node, attrs in G.nodes(data=True):
            if attrs["type"] == "generated":
                labels[node] = "Generated Text"
            else:
                labels[node] = attrs["label"]
                
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold")
        
        # Add text content as annotations
        for node, (x, y) in pos.items():
            text = G.nodes[node]["text"]
            plt.annotate(text, xy=(x, y), xytext=(0, -20), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                        ha="center", fontsize=8)
        
        plt.title("Attribution Trust Path")
        plt.axis("off")
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved network trust path visualization to {output_path}")
        
        return output_path

class HeatmapTrustPathVisualizer(TrustPathVisualizer):
    """
    Visualizer that shows trust paths as a heatmap of token attributions.
    """
    
    def __init__(
        self,
        source_metadata: Optional[Dict[int, Dict[str, str]]] = None,
        output_dir: str = "figures",
        figsize: Tuple[int, int] = (15, 10),
        tokenizer = None
    ):
        """
        Initialize the heatmap trust path visualizer.
        
        Args:
            source_metadata: Metadata about sources
            output_dir: Directory to save visualizations
            figsize: Figure size in inches
            tokenizer: Tokenizer for consistent tokenization (optional)
        """
        super().__init__(source_metadata, output_dir)
        self.figsize = figsize
        self.tokenizer = tokenizer
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if self.tokenizer:
            # Use provided tokenizer
            encoded = self.tokenizer(text, return_offsets_mapping=True)
            tokens = self.tokenizer.convert_ids_to_tokens(encoded.input_ids)
            return tokens
        else:
            # Simple tokenization by whitespace and punctuation
            return re.findall(r'\b\w+\b|[^\w\s]', text)
    
    def visualize_trust_path(
        self,
        generated_text: str,
        source_texts: List[str],
        attributions: Dict[str, Any],
        token_scores: Optional[List[List[float]]] = None,
        filename: str = "heatmap_trust_path.png"
    ) -> str:
        """
        Visualize the trust path as a heatmap.
        
        Args:
            generated_text: Generated text
            source_texts: List of source texts
            attributions: Attribution information
            token_scores: Token-level attribution scores (if available)
            filename: Output filename
            
        Returns:
            Path to the saved visualization
        """
        # Get attributions
        sources = attributions.get("sources", [])
        scores = attributions.get("scores", [])
        
        # Normalize scores if not provided
        if not scores and sources:
            scores = [1.0 / len(sources)] * len(sources)
        
        # Filter to valid sources
        valid_sources = []
        valid_scores = []
        
        for i, source_id in enumerate(sources):
            if source_id < len(source_texts):
                valid_sources.append(source_id)
                valid_scores.append(scores[i] if i < len(scores) else 0.5)
        
        # If no valid sources, create empty visualization
        if not valid_sources:
            plt.figure(figsize=self.figsize)
            plt.text(0.5, 0.5, "No valid attributions", 
                    ha="center", va="center", fontsize=20)
            plt.axis("off")
            output_path = os.path.join(self.output_dir, filename)
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            logger.info(f"Saved empty heatmap visualization to {output_path}")
            return output_path
        
        # Tokenize texts
        generated_tokens = self._tokenize_text(generated_text)
        source_tokens_list = [self._tokenize_text(source_texts[src]) for src in valid_sources]
        
        # Create figure with subplots: one for generated text, one per source
        n_sources = len(valid_sources)
        fig, axs = plt.subplots(n_sources + 1, 1, figsize=self.figsize, 
                                gridspec_kw={"height_ratios": [1.5] + [1] * n_sources})
        
        # Plot generated text
        ax_gen = axs[0]
        
        # If token-level scores provided, use them for coloring
        if token_scores:
            # Check format
            if isinstance(token_scores, list) and len(token_scores) == len(valid_sources):
                # Create combined token scores
                combined_scores = np.zeros(len(generated_tokens))
                
                for i, scores in enumerate(token_scores):
                    # Skip if wrong length
                    if len(scores) != len(generated_tokens):
                        continue
                        
                    # Weight by attribution score
                    weighted_scores = np.array(scores) * valid_scores[i]
                    combined_scores += weighted_scores
                
                # Normalize
                if combined_scores.max() > 0:
                    combined_scores = combined_scores / combined_scores.max()
                
                # Plot as heatmap
                token_positions = np.arange(len(generated_tokens))
                
                # Create a colormap from white to blue
                cmap = LinearSegmentedColormap.from_list(
                    "attribution_cmap", ["white", "skyblue", "dodgerblue"]
                )
                
                ax_gen.bar(token_positions, 1, color=cmap(combined_scores), width=0.8)
                ax_gen.set_xticks(token_positions)
                ax_gen.set_xticklabels(generated_tokens, rotation=45, ha="right", fontsize=8)
                ax_gen.set_yticks([])
        else:
            # Simple text display without coloring
            ax_gen.text(0.5, 0.5, " ".join(generated_tokens), 
                      ha="center", va="center", wrap=True, fontsize=12)
            ax_gen.set_xticks([])
            ax_gen.set_yticks([])
        
        ax_gen.set_title("Generated Text")
        ax_gen.set_xlim(-0.5, len(generated_tokens) - 0.5)
        
        # Plot source texts
        for i, source_id in enumerate(valid_sources):
            ax_src = axs[i + 1]
            source_tokens = source_tokens_list[i]
            
            # Simple text display
            ax_src.text(0.5, 0.5, " ".join(source_tokens), 
                      ha="center", va="center", wrap=True, fontsize=10)
            
            # Add source info
            source_label = self._get_source_label(source_id)
            attribution_score = valid_scores[i]
            ax_src.set_title(f"{source_label} (Attribution Score: {attribution_score:.2f})")
            
            ax_src.set_xticks([])
            ax_src.set_yticks([])
        
        plt.suptitle("Attribution Trust Path", fontsize=16)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved heatmap trust path visualization to {output_path}")
        
        return output_path

class HTMLTrustPathVisualizer(TrustPathVisualizer):
    """
    Visualizer that creates an interactive HTML representation of the trust path.
    """
    
    def __init__(
        self,
        source_metadata: Optional[Dict[int, Dict[str, str]]] = None,
        output_dir: str = "figures"
    ):
        """
        Initialize the HTML trust path visualizer.
        
        Args:
            source_metadata: Metadata about sources
            output_dir: Directory to save visualizations
        """
        super().__init__(source_metadata, output_dir)
    
    def _generate_html(
        self,
        generated_text: str,
        source_texts: List[str],
        attributions: Dict[str, Any],
        token_scores: Optional[List[List[float]]] = None
    ) -> str:
        """
        Generate HTML representation of the trust path.
        
        Args:
            generated_text: Generated text
            source_texts: List of source texts
            attributions: Attribution information
            token_scores: Token-level attribution scores (if available)
            
        Returns:
            HTML string
        """
        # Get attributions
        sources = attributions.get("sources", [])
        scores = attributions.get("scores", [])
        
        # Normalize scores if not provided
        if not scores and sources:
            scores = [1.0 / len(sources)] * len(sources)
        
        # Filter to valid sources
        valid_sources = []
        valid_scores = []
        valid_source_texts = []
        
        for i, source_id in enumerate(sources):
            if source_id < len(source_texts):
                valid_sources.append(source_id)
                valid_scores.append(scores[i] if i < len(scores) else 0.5)
                valid_source_texts.append(source_texts[source_id])
        
        # Start HTML
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Attribution Trust Path</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .generated-text { 
                    background-color: #f0f8ff; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin-bottom: 20px;
                    font-size: 16px;
                    line-height: 1.5;
                }
                .source-container {
                    margin-bottom: 30px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                }
                .source-header {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                    font-weight: bold;
                    background-color: #f5f5f5;
                    padding: 10px;
                    border-radius: 3px;
                }
                .source-text {
                    padding: 10px;
                    font-size: 14px;
                    line-height: 1.4;
                }
                .attribution-score {
                    color: #0066cc;
                }
                .highlighted {
                    background-color: #ffff99;
                }
                h1 { color: #333; }
                h2 { color: #555; margin-top: 30px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Attribution Trust Path</h1>
        """
        
        # Add generated text
        html += f"""
                <h2>Generated Text</h2>
                <div class="generated-text">
                    {generated_text}
                </div>
        """
        
        # Add sources
        if valid_sources:
            html += """
                <h2>Source Attributions</h2>
            """
            
            for i, source_id in enumerate(valid_sources):
                source_text = valid_source_texts[i]
                source_label = self._get_source_label(source_id)
                attribution_score = valid_scores[i]
                
                html += f"""
                <div class="source-container">
                    <div class="source-header">
                        <div>{source_label}</div>
                        <div class="attribution-score">Attribution Score: {attribution_score:.2f}</div>
                    </div>
                    <div class="source-text">
                        {source_text}
                    </div>
                </div>
                """
        else:
            html += """
                <h2>No Valid Attributions Found</h2>
            """
        
        # Close HTML
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def visualize_trust_path(
        self,
        generated_text: str,
        source_texts: List[str],
        attributions: Dict[str, Any],
        token_scores: Optional[List[List[float]]] = None,
        filename: str = "trust_path.html"
    ) -> str:
        """
        Create an HTML visualization of the trust path.
        
        Args:
            generated_text: Generated text
            source_texts: List of source texts
            attributions: Attribution information
            token_scores: Token-level attribution scores (if available)
            filename: Output filename
            
        Returns:
            Path to the saved visualization
        """
        # Generate HTML
        html = self._generate_html(
            generated_text, source_texts, attributions, token_scores
        )
        
        # Save to file
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(html)
        
        logger.info(f"Saved HTML trust path visualization to {output_path}")
        
        return output_path

def generate_comparison_visualization(
    generated_text: str,
    model_results: Dict[str, Dict[str, Any]],
    source_texts: List[str],
    output_dir: str = "figures",
    filename: str = "model_comparison.png"
) -> str:
    """
    Generate a comparison visualization of different attribution methods.
    
    Args:
        generated_text: Generated text
        model_results: Dictionary mapping model names to attribution results
        source_texts: List of source texts
        output_dir: Directory to save visualization
        filename: Output filename
        
    Returns:
        Path to the saved visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up figure
    fig, axs = plt.subplots(len(model_results) + 1, 1, figsize=(12, 3 * (len(model_results) + 1)),
                           gridspec_kw={"height_ratios": [1] + [1] * len(model_results)})
    
    # Display generated text
    axs[0].text(0.5, 0.5, generated_text, ha="center", va="center", wrap=True)
    axs[0].set_title("Generated Text")
    axs[0].axis("off")
    
    # Display attributions for each model
    for i, (model_name, results) in enumerate(model_results.items()):
        ax = axs[i + 1]
        
        # Get sources and scores
        sources = results.get("sources", [])
        scores = results.get("scores", [])
        
        # Normalize scores if not provided
        if not scores and sources:
            scores = [1.0 / len(sources)] * len(sources)
        
        # Filter to valid sources with texts
        valid_sources = []
        valid_scores = []
        
        for j, source_id in enumerate(sources):
            if source_id < len(source_texts):
                valid_sources.append(source_id)
                valid_scores.append(scores[j] if j < len(scores) else 0.5)
        
        # Plot bar chart of attribution scores
        if valid_sources:
            x = np.arange(len(valid_sources))
            ax.bar(x, valid_scores, color="skyblue")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Source {s}" for s in valid_sources])
            ax.set_ylabel("Attribution Score")
            ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, "No valid attributions", ha="center", va="center")
            ax.axis("off")
        
        ax.set_title(f"{model_name} Attribution")
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved model comparison visualization to {output_path}")
    
    return output_path

def generate_attribution_flow_visualization(
    generations: List[str],
    source_texts: List[str],
    attributions: List[Dict[str, Any]],
    output_dir: str = "figures",
    filename: str = "attribution_flow.png"
) -> str:
    """
    Generate a visualization showing the flow of attributions across generations.
    
    Args:
        generations: List of generated texts
        source_texts: List of source texts
        attributions: List of attribution results for each generation
        output_dir: Directory to save visualization
        filename: Output filename
        
    Returns:
        Path to the saved visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add source nodes
    for i, text in enumerate(source_texts):
        G.add_node(f"source_{i}", text=text[:50] + "..." if len(text) > 50 else text, 
                  type="source")
    
    # Add generation nodes and connect to sources
    for i, (gen_text, attr) in enumerate(zip(generations, attributions)):
        gen_node = f"gen_{i}"
        G.add_node(gen_node, text=gen_text[:50] + "..." if len(gen_text) > 50 else gen_text, 
                  type="generation")
        
        # Connect to sources
        sources = attr.get("sources", [])
        scores = attr.get("scores", [])
        
        # Normalize scores if not provided
        if not scores and sources:
            scores = [1.0 / len(sources)] * len(sources)
        
        for j, source_id in enumerate(sources):
            if source_id < len(source_texts):
                score = scores[j] if j < len(scores) else 0.5
                G.add_edge(f"source_{source_id}", gen_node, weight=score)
    
    # Draw the graph
    plt.figure(figsize=(12, 10))
    
    # Use layout that separates sources and generations
    pos = {}
    
    # Position sources on the left
    source_nodes = [n for n in G.nodes() if "source" in n]
    for i, node in enumerate(source_nodes):
        pos[node] = (-1, (i - len(source_nodes)/2) * 2)
    
    # Position generations on the right
    gen_nodes = [n for n in G.nodes() if "gen" in n]
    for i, node in enumerate(gen_nodes):
        pos[node] = (1, (i - len(gen_nodes)/2) * 2)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=source_nodes, 
                          node_color="lightgreen", 
                          node_size=1500, 
                          alpha=0.8)
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=gen_nodes, 
                          node_color="skyblue", 
                          node_size=1500, 
                          alpha=0.8)
    
    # Draw edges with weights
    edges = G.edges(data=True)
    edge_weights = [e[2]["weight"] * 3 for e in edges]
    
    nx.draw_networkx_edges(G, pos, 
                          width=edge_weights, 
                          alpha=0.7, 
                          edge_color="gray", 
                          arrows=True)
    
    # Draw labels
    labels = {}
    for node in G.nodes():
        if "source" in node:
            source_id = int(node.split("_")[1])
            labels[node] = f"Source {source_id}"
        else:
            gen_id = int(node.split("_")[1])
            labels[node] = f"Generation {gen_id+1}"
    
    nx.draw_networkx_labels(G, pos, labels=labels)
    
    # Add node texts as annotations
    for node, (x, y) in pos.items():
        text = G.nodes[node]["text"]
        plt.annotate(text, xy=(x, y), xytext=(0, -30), 
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
                    ha="center", fontsize=8)
    
    plt.title("Attribution Flow Across Generations")
    plt.axis("off")
    
    # Save figure
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved attribution flow visualization to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Test the visualizers
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    generated_text = "The quick brown fox jumps over the lazy dog. Python is a high-level programming language."
    
    source_texts = [
        "The quick brown fox jumps over the lazy dog. This sentence contains all the letters of the English alphabet.",
        "Python is a high-level, interpreted programming language known for its readability and simplicity.",
        "Machine learning is a subset of artificial intelligence that provides systems the ability to learn from data."
    ]
    
    attributions = {
        "sources": [0, 1],
        "scores": [0.9, 0.7]
    }
    
    source_metadata = {
        0: {
            "title": "Pangram Example",
            "author": "Anonymous"
        },
        1: {
            "title": "Python Overview",
            "author": "Python Foundation"
        },
        2: {
            "title": "ML Basics",
            "author": "AI Researcher"
        }
    }
    
    # Test NetworkTrustPathVisualizer
    print("Testing NetworkTrustPathVisualizer...")
    
    network_viz = NetworkTrustPathVisualizer(source_metadata)
    network_path = network_viz.visualize_trust_path(
        generated_text, source_texts, attributions, "test_network.png"
    )
    
    print(f"Network visualization saved to {network_path}")
    
    # Test HeatmapTrustPathVisualizer
    print("\nTesting HeatmapTrustPathVisualizer...")
    
    heatmap_viz = HeatmapTrustPathVisualizer(source_metadata)
    heatmap_path = heatmap_viz.visualize_trust_path(
        generated_text, source_texts, attributions, "test_heatmap.png"
    )
    
    print(f"Heatmap visualization saved to {heatmap_path}")
    
    # Test HTMLTrustPathVisualizer
    print("\nTesting HTMLTrustPathVisualizer...")
    
    html_viz = HTMLTrustPathVisualizer(source_metadata)
    html_path = html_viz.visualize_trust_path(
        generated_text, source_texts, attributions, "test_html.html"
    )
    
    print(f"HTML visualization saved to {html_path}")
    
    # Test comparison visualization
    print("\nTesting comparison visualization...")
    
    model_results = {
        "AGT": {
            "sources": [0, 1],
            "scores": [0.9, 0.7]
        },
        "Post-hoc": {
            "sources": [0],
            "scores": [0.8]
        },
        "Data Shapley": {
            "sources": [0, 1, 2],
            "scores": [0.6, 0.5, 0.3]
        }
    }
    
    comparison_path = generate_comparison_visualization(
        generated_text, model_results, source_texts, filename="test_comparison.png"
    )
    
    print(f"Comparison visualization saved to {comparison_path}")
    
    # Test attribution flow visualization
    print("\nTesting attribution flow visualization...")
    
    generations = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a high-level programming language.",
        "ML systems learn from data without explicit programming."
    ]
    
    gen_attributions = [
        {"sources": [0], "scores": [0.9]},
        {"sources": [1], "scores": [0.8]},
        {"sources": [2], "scores": [0.7]}
    ]
    
    flow_path = generate_attribution_flow_visualization(
        generations, source_texts, gen_attributions, filename="test_flow.png"
    )
    
    print(f"Flow visualization saved to {flow_path}")
    
    print("\nAll visualization tests completed!")