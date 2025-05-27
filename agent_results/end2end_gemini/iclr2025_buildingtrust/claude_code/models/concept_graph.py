"""
Concept Graph Module

This module implements the construction, analysis, and visualization of concept graphs.
"""

import os
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from matplotlib.colors import to_rgba

logger = logging.getLogger(__name__)

class ConceptGraph:
    """
    Constructs and visualizes concept graphs from identified concepts.
    
    This class provides functionality for:
    1. Building a directed graph from concepts and their relationships
    2. Analyzing the graph structure and properties
    3. Visualizing the concept graph
    4. Extracting insights from the graph
    """
    
    def __init__(self):
        """Initialize the ConceptGraph."""
        self.graph = nx.DiGraph()  # Directed graph for concepts
        self.concept_info = {}     # Store information about concepts
    
    def build_from_concepts(
        self,
        concepts: Dict[str, Dict[str, Any]],
        attention_weights: Optional[Dict[int, Dict[int, List[np.ndarray]]]] = None,
        temporal_ordering: bool = True,
        min_edge_weight: float = 0.1
    ) -> nx.DiGraph:
        """
        Build a concept graph from identified concepts.
        
        Args:
            concepts: Dictionary of concepts with their properties
            attention_weights: Optional dictionary of attention weights
            temporal_ordering: Whether to use temporal ordering for edge creation
            min_edge_weight: Minimum edge weight threshold
            
        Returns:
            Constructed NetworkX directed graph
        """
        logger.info(f"Building concept graph from {len(concepts)} concepts")
        
        # Create a new directed graph
        self.graph = nx.DiGraph()
        
        # Store concept information
        self.concept_info = concepts
        
        # Add concept nodes
        for concept_name, concept_data in concepts.items():
            # Extract node attributes
            node_attrs = {
                'size': concept_data.get('size', 0),
                'position_range': concept_data.get('position_range', (0, 0))
            }
            
            # Add centroid if available
            if 'centroid' in concept_data:
                node_attrs['centroid'] = concept_data['centroid']
            
            # Add label if available
            if 'concept_label' in concept_data:
                node_attrs['concept_label'] = concept_data['concept_label']
            
            # Add the node
            self.graph.add_node(concept_name, **node_attrs)
        
        # Add edges based on temporal ordering
        if temporal_ordering:
            # Sort concepts by their starting position
            sorted_concepts = sorted(
                concepts.items(),
                key=lambda x: x[1].get('position_range', (0, 0))[0]
            )
            
            # Create edges between consecutive concepts
            for i in range(len(sorted_concepts) - 1):
                src_concept, src_data = sorted_concepts[i]
                dst_concept, dst_data = sorted_concepts[i + 1]
                
                src_end = src_data.get('position_range', (0, 0))[1]
                dst_start = dst_data.get('position_range', (0, 0))[0]
                
                # Only add edge if the concepts are connected temporally
                if src_end >= dst_start:
                    self.graph.add_edge(
                        src_concept,
                        dst_concept,
                        weight=1.0,
                        type='temporal'
                    )
        
        # Add edges based on attention weights if provided
        if attention_weights:
            # Create a mapping from token positions to concepts
            position_to_concept = {}
            for concept_name, concept_data in concepts.items():
                for pos in concept_data.get('token_positions', []):
                    if pos not in position_to_concept:
                        position_to_concept[pos] = []
                    position_to_concept[pos].append(concept_name)
            
            # Aggregate attention across layers and heads
            agg_attention = None
            for layer_idx, layer_attn in attention_weights.items():
                for head_idx, head_attn in layer_attn.items():
                    # Stack the attention weights for this layer and head
                    attn_stack = np.stack(head_attn)
                    
                    if agg_attention is None:
                        agg_attention = attn_stack
                    else:
                        # Ensure compatible shapes
                        min_tokens = min(agg_attention.shape[1], attn_stack.shape[1])
                        agg_attention = agg_attention[:, :min_tokens]
                        attn_stack = attn_stack[:, :min_tokens]
                        
                        # Average the attention
                        agg_attention = (agg_attention + attn_stack) / 2
            
            if agg_attention is not None:
                # Create edges based on attention
                for src_pos, src_concepts in position_to_concept.items():
                    for dst_pos, dst_concepts in position_to_concept.items():
                        if src_pos < dst_pos:  # Only consider forward attention
                            # Get the attention weight from source to destination
                            if src_pos < agg_attention.shape[0] and dst_pos < agg_attention.shape[1]:
                                attn_weight = agg_attention[src_pos, dst_pos]
                                
                                # If weight exceeds threshold, add edges between concepts
                                if attn_weight > min_edge_weight:
                                    for src_concept in src_concepts:
                                        for dst_concept in dst_concepts:
                                            if self.graph.has_edge(src_concept, dst_concept):
                                                # Update edge weight
                                                current_weight = self.graph[src_concept][dst_concept]['weight']
                                                self.graph[src_concept][dst_concept]['weight'] = max(current_weight, attn_weight)
                                                self.graph[src_concept][dst_concept]['type'] = 'attention'
                                            else:
                                                # Add new edge
                                                self.graph.add_edge(
                                                    src_concept,
                                                    dst_concept,
                                                    weight=attn_weight,
                                                    type='attention'
                                                )
        
        logger.info(f"Graph construction complete with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def build_from_labeled_clusters(
        self,
        cluster_result: Dict[str, Any],
        min_edge_weight: float = 0.1
    ) -> nx.DiGraph:
        """
        Build a concept graph from labeled clusters.
        
        Args:
            cluster_result: Result from concept mapping with labeled clusters
            min_edge_weight: Minimum edge weight threshold
            
        Returns:
            Constructed NetworkX directed graph
        """
        logger.info("Building concept graph from labeled clusters")
        
        # Create a new directed graph
        self.graph = nx.DiGraph()
        
        # Extract labeled clusters
        labeled_clusters = cluster_result.get('labeled_clusters', {})
        
        if not labeled_clusters:
            logger.warning("No labeled clusters found in the input")
            return self.graph
        
        # Store labeled clusters as concept info
        self.concept_info = labeled_clusters
        
        # Add nodes for each labeled cluster
        for cluster_id, cluster_data in labeled_clusters.items():
            # Get the concept label or use a default
            concept_label = cluster_data.get('concept_label', f"Cluster {cluster_id}")
            
            # Extract node attributes
            node_attrs = {
                'size': cluster_data.get('size', 0),
                'position_range': cluster_data.get('position_range', (0, 0)),
                'concept_label': concept_label,
                'cluster_id': cluster_id
            }
            
            # Add the node with the concept label as node ID
            self.graph.add_node(concept_label, **node_attrs)
        
        # Add edges based on temporal ordering
        # Sort clusters by their starting position
        sorted_clusters = sorted(
            [(concept_label, cluster_data) for cluster_id, cluster_data in labeled_clusters.items()],
            key=lambda x: x[1].get('position_range', (0, 0))[0]
        )
        
        # Create edges between consecutive clusters
        for i in range(len(sorted_clusters) - 1):
            src_concept, src_data = sorted_clusters[i]
            dst_concept, dst_data = sorted_clusters[i + 1]
            
            src_end = src_data.get('position_range', (0, 0))[1]
            dst_start = dst_data.get('position_range', (0, 0))[0]
            
            # Calculate a weight based on temporal overlap
            if src_end >= dst_start:
                overlap = min(src_end, dst_start) - max(src_data.get('position_range', (0, 0))[0], dst_data.get('position_range', (0, 0))[0])
                overlap = max(0, overlap)
                
                weight = overlap / (src_end - src_data.get('position_range', (0, 0))[0] + 1)
                
                if weight > min_edge_weight:
                    self.graph.add_edge(
                        src_concept,
                        dst_concept,
                        weight=weight,
                        type='temporal'
                    )
        
        logger.info(f"Graph construction complete with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def analyze_graph_structure(self) -> Dict[str, Any]:
        """
        Analyze the structure of the concept graph.
        
        Returns:
            Dictionary of graph analysis metrics
        """
        logger.info("Analyzing concept graph structure")
        
        analysis = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_dag': nx.is_directed_acyclic_graph(self.graph)
        }
        
        # Compute centrality measures
        try:
            analysis['degree_centrality'] = nx.degree_centrality(self.graph)
            analysis['in_degree_centrality'] = nx.in_degree_centrality(self.graph)
            analysis['out_degree_centrality'] = nx.out_degree_centrality(self.graph)
            
            # Only compute betweenness centrality if there are enough nodes
            if self.graph.number_of_nodes() > 2:
                analysis['betweenness_centrality'] = nx.betweenness_centrality(self.graph)
            
            # Identify critical concepts (high centrality)
            if 'betweenness_centrality' in analysis:
                top_betweenness = sorted(
                    analysis['betweenness_centrality'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                analysis['critical_concepts'] = [node for node, _ in top_betweenness]
            else:
                # Use degree centrality if betweenness is not available
                top_degree = sorted(
                    analysis['degree_centrality'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                analysis['critical_concepts'] = [node for node, _ in top_degree]
            
            # Identify source and sink concepts
            sources = [node for node, in_deg in self.graph.in_degree() if in_deg == 0]
            sinks = [node for node, out_deg in self.graph.out_degree() if out_deg == 0]
            
            analysis['source_concepts'] = sources
            analysis['sink_concepts'] = sinks
            
            # Compute paths
            if analysis['is_dag']:
                # Find all paths from sources to sinks
                all_paths = []
                for source in sources:
                    for sink in sinks:
                        try:
                            paths = list(nx.all_simple_paths(self.graph, source, sink))
                            all_paths.extend(paths)
                        except:
                            pass
                
                analysis['num_paths'] = len(all_paths)
                
                if all_paths:
                    analysis['avg_path_length'] = sum(len(path) for path in all_paths) / len(all_paths)
                    analysis['longest_path'] = max(all_paths, key=len)
                    
                    # Count path frequencies for concepts
                    concept_path_counts = {}
                    for path in all_paths:
                        for concept in path:
                            concept_path_counts[concept] = concept_path_counts.get(concept, 0) + 1
                    
                    analysis['concept_path_frequency'] = concept_path_counts
        except Exception as e:
            logger.error(f"Error computing graph metrics: {str(e)}")
        
        return analysis
    
    def visualize_graph(
        self,
        save_path: Optional[str] = None,
        layout: str = 'temporal',
        node_size_factor: float = 300.0,
        edge_width_factor: float = 2.0,
        figsize: Tuple[int, int] = (14, 10),
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the concept graph.
        
        Args:
            save_path: Path to save the visualization
            layout: Graph layout ('temporal', 'spring', 'kamada_kawai', 'planar')
            node_size_factor: Scaling factor for node sizes
            edge_width_factor: Scaling factor for edge widths
            figsize: Figure size
            title: Optional title for the visualization
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Visualizing concept graph with {layout} layout")
        
        if not self.graph.nodes():
            logger.warning("Cannot visualize empty graph")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "Empty Graph", horizontalalignment='center', fontsize=20)
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine node positions based on layout
        if layout == 'temporal':
            # Use position_range to determine x-coordinate
            pos = {}
            y_values = {}
            
            # First, determine y levels to minimize edge crossings
            # Simple algorithm: place nodes in order of appearance
            sorted_nodes = sorted(
                self.graph.nodes(data=True),
                key=lambda x: x[1].get('position_range', (0, 0))[0]
            )
            
            used_levels = set()
            
            for node, data in sorted_nodes:
                pos_range = data.get('position_range', (0, 0))
                x_pos = (pos_range[0] + pos_range[1]) / 2
                
                # Find suitable y level
                level = 0
                while level in used_levels:
                    level += 1
                
                used_levels.add(level)
                y_values[node] = level
                
                pos[node] = (x_pos, level)
        
        elif layout == 'spring':
            pos = nx.spring_layout(self.graph, seed=42)
        
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph)
        
        elif layout == 'planar':
            try:
                pos = nx.planar_layout(self.graph)
            except:
                logger.warning("Graph is not planar, falling back to spring layout")
                pos = nx.spring_layout(self.graph, seed=42)
        
        else:
            logger.warning(f"Unknown layout '{layout}', using spring layout")
            pos = nx.spring_layout(self.graph, seed=42)
        
        # Get node sizes based on concept data
        node_sizes = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            size = node_data.get('size', 1)
            node_sizes.append(size * node_size_factor)
        
        # Get edge widths based on weights
        edge_widths = []
        for _, _, data in self.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            edge_widths.append(weight * edge_width_factor)
        
        # Create colormap for the edges
        edge_colors = []
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'temporal')
            if edge_type == 'temporal':
                edge_colors.append('blue')
            elif edge_type == 'attention':
                edge_colors.append('red')
            else:
                edge_colors.append('gray')
        
        # Draw nodes and edges
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_size=node_sizes,
            node_color='lightblue',
            alpha=0.8,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            self.graph,
            pos,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.7,
            arrows=True,
            arrowsize=15,
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
        
        # Draw labels with appropriate size
        node_labels = {}
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            label = node_data.get('concept_label', node)
            node_labels[node] = label
        
        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels=node_labels,
            font_size=10,
            font_family='sans-serif',
            font_weight='bold',
            ax=ax
        )
        
        # Set title
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title(f"Concept Graph ({self.graph.number_of_nodes()} concepts, {self.graph.number_of_edges()} connections)", fontsize=16)
        
        # Remove axis and ticks
        plt.axis('off')
        
        # Add legend
        temporal_patch = plt.Line2D([0], [0], color='blue', linewidth=2, label='Temporal Connection')
        attention_patch = plt.Line2D([0], [0], color='red', linewidth=2, label='Attention Connection')
        
        plt.legend(handles=[temporal_patch, attention_patch], loc='upper right')
        
        plt.tight_layout()
        
        # Save the figure if save_path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved concept graph visualization to {save_path}")
        
        return fig
    
    def extract_reasoning_path(self) -> List[str]:
        """
        Extract the main reasoning path from the concept graph.
        
        Returns:
            List of concepts in the main reasoning path
        """
        logger.info("Extracting main reasoning path from concept graph")
        
        if not self.graph.nodes():
            logger.warning("Cannot extract path from empty graph")
            return []
        
        # Check if the graph is a DAG
        if not nx.is_directed_acyclic_graph(self.graph):
            logger.warning("Graph is not a DAG. Extracting longest path may not be meaningful.")
        
        # Find source and sink nodes
        sources = [node for node, in_deg in self.graph.in_degree() if in_deg == 0]
        sinks = [node for node, out_deg in self.graph.out_degree() if out_deg == 0]
        
        # If there are no sources or sinks, use nodes with min in-degree or max out-degree
        if not sources:
            in_degrees = dict(self.graph.in_degree())
            min_in_degree = min(in_degrees.values())
            sources = [node for node, deg in in_degrees.items() if deg == min_in_degree]
        
        if not sinks:
            out_degrees = dict(self.graph.out_degree())
            min_out_degree = min(out_degrees.values())
            sinks = [node for node, deg in out_degrees.items() if deg == min_out_degree]
        
        # Find all paths from sources to sinks
        all_paths = []
        for source in sources:
            for sink in sinks:
                try:
                    paths = list(nx.all_simple_paths(self.graph, source, sink))
                    all_paths.extend(paths)
                except:
                    continue
        
        if not all_paths:
            logger.warning("No paths found from sources to sinks")
            
            # Fall back to topological sort
            try:
                return list(nx.topological_sort(self.graph))
            except:
                logger.error("Failed to compute topological sort")
                return list(self.graph.nodes())
        
        # Find the longest path
        longest_path = max(all_paths, key=len)
        
        return longest_path
    
    def evaluate_concept_graph_quality(
        self,
        reference_steps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of the concept graph.
        
        Args:
            reference_steps: Optional reference reasoning steps to compare against
            
        Returns:
            Dictionary of quality metrics
        """
        logger.info("Evaluating concept graph quality")
        
        metrics = {
            'num_concepts': self.graph.number_of_nodes(),
            'num_connections': self.graph.number_of_edges(),
            'graph_density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'is_dag': nx.is_directed_acyclic_graph(self.graph)
        }
        
        # Extract the main reasoning path
        main_path = self.extract_reasoning_path()
        metrics['reasoning_path_length'] = len(main_path)
        metrics['reasoning_path'] = main_path
        
        # Compare with reference steps if provided
        if reference_steps:
            metrics['reference_steps_length'] = len(reference_steps)
            
            # Simple length comparison
            metrics['length_ratio'] = len(main_path) / len(reference_steps) if reference_steps else 0
            
            # TODO: Implement more sophisticated comparison metrics like BLEU or semantic similarity
            # This would require embedding both the concept labels and reference steps
        
        return metrics
    
    def generate_graph_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the concept graph.
        
        Returns:
            Dictionary containing graph summary information
        """
        logger.info("Generating concept graph summary")
        
        # Extract various metrics and information
        analysis = self.analyze_graph_structure()
        main_path = self.extract_reasoning_path()
        
        # Calculate degree distributions
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        
        # Get information about each node
        node_info = {}
        for node, data in self.graph.nodes(data=True):
            node_info[node] = {
                'size': data.get('size', 0),
                'position_range': data.get('position_range', (0, 0)),
                'in_degree': in_degrees.get(node, 0),
                'out_degree': out_degrees.get(node, 0),
                'in_main_path': node in main_path,
                'centrality': analysis.get('degree_centrality', {}).get(node, 0)
            }
        
        # Prepare summary
        summary = {
            'graph_metrics': {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'density': analysis.get('density', 0),
                'is_dag': analysis.get('is_dag', False),
                'avg_in_degree': sum(in_degrees.values()) / len(in_degrees) if in_degrees else 0,
                'avg_out_degree': sum(out_degrees.values()) / len(out_degrees) if out_degrees else 0
            },
            'critical_concepts': analysis.get('critical_concepts', []),
            'source_concepts': analysis.get('source_concepts', []),
            'sink_concepts': analysis.get('sink_concepts', []),
            'main_reasoning_path': main_path,
            'node_info': node_info
        }
        
        return summary
    
    def get_dot_representation(self) -> str:
        """
        Generate a DOT representation of the graph for visualization with GraphViz.
        
        Returns:
            DOT representation as a string
        """
        dot = ["digraph ConceptGraph {"]
        dot.append("  rankdir=LR;")  # Left to right layout
        dot.append("  node [shape=box, style=filled, fillcolor=lightblue];")
        
        # Add nodes
        for node, data in self.graph.nodes(data=True):
            label = data.get('concept_label', node)
            size = data.get('size', 1)
            
            # Calculate node size based on concept size
            width = 1.0 + 0.1 * size
            height = 0.75 + 0.05 * size
            
            dot.append(f'  "{node}" [label="{label}", width={width}, height={height}];')
        
        # Add edges
        for src, dst, data in self.graph.edges(data=True):
            weight = data.get('weight', 1.0)
            edge_type = data.get('type', 'temporal')
            
            # Set color based on edge type
            if edge_type == 'temporal':
                color = "blue"
            elif edge_type == 'attention':
                color = "red"
            else:
                color = "gray"
            
            # Set penwidth based on weight
            penwidth = 1.0 + weight * 2
            
            dot.append(f'  "{src}" -> "{dst}" [color="{color}", penwidth={penwidth}];')
        
        dot.append("}")
        
        return "\n".join(dot)