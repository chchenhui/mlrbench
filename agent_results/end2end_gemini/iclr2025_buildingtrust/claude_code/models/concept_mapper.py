"""
Concept Mapper Module

This module implements concept identification and mapping from LLM internal states.
"""

import os
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import matplotlib.pyplot as plt
import umap.umap_ as umap

logger = logging.getLogger(__name__)

class ConceptMapper:
    """
    Maps internal LLM states to human-understandable concepts.
    
    This class provides functionality for:
    1. Concept discovery through unsupervised clustering
    2. Semi-supervised concept anchoring with predefined templates
    3. LLM-aided concept labeling using external models
    """
    
    def __init__(
        self,
        use_openai_for_labeling: bool = True,
        openai_model: str = "gpt-4o-mini",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the ConceptMapper.
        
        Args:
            use_openai_for_labeling: Whether to use OpenAI API for concept labeling
            openai_model: OpenAI model to use for concept labeling
            cache_dir: Directory to cache concept embeddings and labels
        """
        self.use_openai_for_labeling = use_openai_for_labeling
        self.openai_model = openai_model
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # OpenAI client for concept labeling
        if use_openai_for_labeling:
            try:
                self.openai_client = OpenAI()
                logger.info(f"Initialized OpenAI client for concept labeling using {openai_model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                self.use_openai_for_labeling = False
        
        # Default predefined concept templates for common reasoning tasks
        self.predefined_concepts = {
            'math_reasoning': [
                'identifying_variables',
                'retrieving_formula',
                'performing_addition',
                'performing_subtraction',
                'performing_multiplication',
                'performing_division',
                'comparing_values',
                'checking_constraints',
                'final_answer_computation'
            ],
            'qa_reasoning': [
                'extracting_key_information',
                'retrieving_context',
                'establishing_relationships',
                'making_inference',
                'evaluating_evidence',
                'drawing_conclusion',
                'formulating_answer'
            ],
            'logical_reasoning': [
                'premise_identification',
                'applying_rule',
                'conditional_reasoning',
                'negation',
                'conjunction',
                'disjunction',
                'implication',
                'contradiction_detection',
                'conclusion_derivation'
            ]
        }
        
        # Initialize PCA and UMAP for dimensionality reduction
        self.pca = None
        self.umap_reducer = None
    
    def discover_concepts_unsupervised(
        self,
        hidden_states: Dict[int, List[torch.Tensor]],
        num_clusters: int = 10,
        pca_components: int = 50,
        umap_components: int = 2,
        clustering_method: str = "kmeans",
        min_samples_per_cluster: int = 3,
        visualize: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Discover concepts through unsupervised clustering of hidden states.
        
        Args:
            hidden_states: Dictionary of hidden states per layer
            num_clusters: Number of clusters (concepts) to identify
            pca_components: Number of PCA components for dimensionality reduction
            umap_components: Number of UMAP components for visualization
            clustering_method: Clustering method to use ("kmeans" or "dbscan")
            min_samples_per_cluster: Minimum samples per cluster (for DBSCAN)
            visualize: Whether to generate visualization of clusters
            save_path: Path to save visualization and results
            
        Returns:
            Dictionary containing discovered concepts and cluster information
        """
        logger.info(f"Discovering concepts using unsupervised clustering with {clustering_method}")
        
        # Flatten hidden states from all layers
        flat_states = []
        token_positions = []
        layer_indices = []
        
        for layer_idx, layer_states in hidden_states.items():
            for pos, state in enumerate(layer_states):
                # Convert state to numpy if it's a tensor
                if isinstance(state, torch.Tensor):
                    if state.dim() > 1:  # Handle batched states (batch, hidden_dim)
                        for batch_idx in range(state.shape[0]):
                            flat_states.append(state[batch_idx].numpy())
                            token_positions.append(pos)
                            layer_indices.append(layer_idx)
                    else:  # Handle single state (hidden_dim)
                        flat_states.append(state.numpy())
                        token_positions.append(pos)
                        layer_indices.append(layer_idx)
                else:  # Already numpy array
                    flat_states.append(state)
                    token_positions.append(pos)
                    layer_indices.append(layer_idx)
        
        # Convert to numpy array
        X = np.array(flat_states)
        logger.info(f"Collected {X.shape[0]} hidden states for clustering")
        
        # Apply PCA for dimensionality reduction
        if pca_components < X.shape[1]:
            logger.info(f"Reducing dimensionality with PCA from {X.shape[1]} to {pca_components}")
            self.pca = PCA(n_components=pca_components, random_state=42)
            X_pca = self.pca.fit_transform(X)
            logger.info(f"PCA explained variance ratio: {sum(self.pca.explained_variance_ratio_):.3f}")
        else:
            logger.info("Skipping PCA as requested components exceed data dimensionality")
            X_pca = X
        
        # Apply UMAP for visualization
        logger.info(f"Applying UMAP for visualization (components={umap_components})")
        self.umap_reducer = umap.UMAP(n_components=umap_components, random_state=42)
        X_umap = self.umap_reducer.fit_transform(X_pca)
        
        # Cluster using the specified method
        if clustering_method == "kmeans":
            logger.info(f"Clustering with KMeans (k={num_clusters})")
            cluster_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = cluster_model.fit_predict(X_pca)
            
            # Calculate cluster centers and distances
            cluster_centers = cluster_model.cluster_centers_
            
            # Calculate distances to cluster centers
            distances = []
            for i, x in enumerate(X_pca):
                cluster_idx = cluster_labels[i]
                center = cluster_centers[cluster_idx]
                distance = np.linalg.norm(x - center)
                distances.append(distance)
            
        elif clustering_method == "dbscan":
            logger.info(f"Clustering with DBSCAN (min_samples={min_samples_per_cluster})")
            # Use a reasonable epsilon based on the data
            distances = []
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min_samples_per_cluster)
            nn.fit(X_pca)
            dists, _ = nn.kneighbors(X_pca)
            distances = np.sort(dists[:, min_samples_per_cluster-1])
            epsilon = np.mean(distances)
            
            logger.info(f"Using epsilon={epsilon:.4f} for DBSCAN")
            cluster_model = DBSCAN(eps=epsilon, min_samples=min_samples_per_cluster)
            cluster_labels = cluster_model.fit_predict(X_pca)
            
            # Handle outliers (label -1) by assigning them to the nearest cluster
            if -1 in cluster_labels:
                outlier_indices = np.where(cluster_labels == -1)[0]
                non_outlier_indices = np.where(cluster_labels != -1)[0]
                
                if len(non_outlier_indices) > 0:
                    # Calculate centroids of non-outlier clusters
                    unique_clusters = np.unique(cluster_labels[non_outlier_indices])
                    centroids = {}
                    
                    for cluster_idx in unique_clusters:
                        cluster_points = X_pca[cluster_labels == cluster_idx]
                        centroids[cluster_idx] = np.mean(cluster_points, axis=0)
                    
                    # Assign outliers to nearest centroid
                    for outlier_idx in outlier_indices:
                        outlier_point = X_pca[outlier_idx]
                        
                        # Calculate distances to all centroids
                        min_dist = float('inf')
                        closest_cluster = 0
                        
                        for cluster_idx, centroid in centroids.items():
                            dist = np.linalg.norm(outlier_point - centroid)
                            if dist < min_dist:
                                min_dist = dist
                                closest_cluster = cluster_idx
                        
                        cluster_labels[outlier_idx] = closest_cluster
                else:
                    # If all points are outliers, just use a simple kmeans
                    logger.warning("All points classified as outliers, falling back to KMeans")
                    cluster_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                    cluster_labels = cluster_model.fit_predict(X_pca)
        
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")
        
        # Collect points for each cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = {
                    'points': [],
                    'positions': [],
                    'layers': [],
                    'umap_points': []
                }
            
            clusters[label]['points'].append(X_pca[i])
            clusters[label]['positions'].append(token_positions[i])
            clusters[label]['layers'].append(layer_indices[i])
            clusters[label]['umap_points'].append(X_umap[i])
        
        # Calculate cluster statistics
        for label, cluster in clusters.items():
            cluster['size'] = len(cluster['points'])
            cluster['center'] = np.mean(cluster['points'], axis=0)
            cluster['position_range'] = (min(cluster['positions']), max(cluster['positions']))
            cluster['common_layers'] = sorted(set(cluster['layers']), key=cluster['layers'].count, reverse=True)
            
            # Find most representative point (closest to center)
            distances_to_center = [np.linalg.norm(p - cluster['center']) for p in cluster['points']]
            cluster['representative_idx'] = np.argmin(distances_to_center)
        
        # Visualize clusters if requested
        if visualize:
            if umap_components == 2:
                self._visualize_clusters_2d(X_umap, cluster_labels, clusters, save_path)
            elif umap_components == 3:
                self._visualize_clusters_3d(X_umap, cluster_labels, clusters, save_path)
        
        # Prepare result
        result = {
            'num_clusters': len(clusters),
            'clusters': clusters,
            'pca': self.pca,
            'umap': self.umap_reducer,
            'cluster_labels': cluster_labels.tolist(),
            'token_positions': token_positions,
            'layer_indices': layer_indices,
            'umap_embedding': X_umap
        }
        
        return result
    
    def semi_supervised_concept_anchoring(
        self,
        hidden_states: Dict[int, List[torch.Tensor]],
        concept_list: List[str],
        reasoning_type: str = 'general',
        pca_components: int = 50,
        visualize: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use semi-supervised concept anchoring to map hidden states to predefined concepts.
        
        Args:
            hidden_states: Dictionary of hidden states per layer
            concept_list: List of concept names to anchor
            reasoning_type: Type of reasoning for default concepts ('math_reasoning', 'qa_reasoning', 'logical_reasoning')
            pca_components: Number of PCA components for dimensionality reduction
            visualize: Whether to generate visualization
            save_path: Path to save visualization and results
            
        Returns:
            Dictionary mapping concepts to states and metrics
        """
        logger.info(f"Performing semi-supervised concept anchoring for {len(concept_list)} concepts")
        
        # If no concepts provided, use predefined ones for the reasoning type
        if not concept_list and reasoning_type in self.predefined_concepts:
            concept_list = self.predefined_concepts[reasoning_type]
            logger.info(f"Using {len(concept_list)} predefined concepts for {reasoning_type}")
        
        # Flatten hidden states from all layers
        flat_states = []
        token_positions = []
        layer_indices = []
        
        for layer_idx, layer_states in hidden_states.items():
            for pos, state in enumerate(layer_states):
                # Convert state to numpy if it's a tensor
                if isinstance(state, torch.Tensor):
                    if state.dim() > 1:  # Handle batched states
                        for batch_idx in range(state.shape[0]):
                            flat_states.append(state[batch_idx].numpy())
                            token_positions.append(pos)
                            layer_indices.append(layer_idx)
                    else:
                        flat_states.append(state.numpy())
                        token_positions.append(pos)
                        layer_indices.append(layer_idx)
                else:
                    flat_states.append(state)
                    token_positions.append(pos)
                    layer_indices.append(layer_idx)
        
        # Convert to numpy array
        X = np.array(flat_states)
        logger.info(f"Collected {X.shape[0]} hidden states for concept anchoring")
        
        # Apply PCA for dimensionality reduction
        if pca_components < X.shape[1]:
            logger.info(f"Reducing dimensionality with PCA from {X.shape[1]} to {pca_components}")
            self.pca = PCA(n_components=pca_components, random_state=42)
            X_pca = self.pca.fit_transform(X)
            logger.info(f"PCA explained variance ratio: {sum(self.pca.explained_variance_ratio_):.3f}")
        else:
            logger.info("Skipping PCA as requested components exceed data dimensionality")
            X_pca = X
        
        # Apply UMAP for visualization
        logger.info("Applying UMAP for visualization (2D)")
        self.umap_reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = self.umap_reducer.fit_transform(X_pca)
        
        # Anchor concepts using temporal position
        # For simplicity, we'll divide the token sequence into segments corresponding to concepts
        sequence_length = max(token_positions) + 1
        segment_size = sequence_length // len(concept_list)
        
        # Initialize concept anchors
        concept_anchors = {}
        for i, concept in enumerate(concept_list):
            start_pos = i * segment_size
            end_pos = (i + 1) * segment_size if i < len(concept_list) - 1 else sequence_length
            
            # Find states in this position range
            concept_indices = [j for j, pos in enumerate(token_positions) if start_pos <= pos < end_pos]
            concept_states = X_pca[concept_indices]
            concept_umap = X_umap[concept_indices]
            
            # Calculate centroid
            if len(concept_states) > 0:
                centroid = np.mean(concept_states, axis=0)
                
                concept_anchors[concept] = {
                    'indices': concept_indices,
                    'states': concept_states,
                    'umap_points': concept_umap,
                    'centroid': centroid,
                    'position_range': (start_pos, end_pos),
                    'token_positions': [token_positions[j] for j in concept_indices],
                    'layer_indices': [layer_indices[j] for j in concept_indices],
                    'size': len(concept_indices)
                }
                
                # Find most representative state (closest to centroid)
                if len(concept_states) > 0:
                    distances = [np.linalg.norm(state - centroid) for state in concept_states]
                    representative_idx = np.argmin(distances)
                    concept_anchors[concept]['representative_idx'] = representative_idx
                    concept_anchors[concept]['representative_state'] = concept_states[representative_idx]
            else:
                logger.warning(f"No states found for concept {concept} in position range {start_pos}-{end_pos}")
        
        # Visualize concept anchors if requested
        if visualize:
            self._visualize_concept_anchors(X_umap, concept_anchors, token_positions, save_path)
        
        # Prepare result
        result = {
            'num_concepts': len(concept_anchors),
            'concept_anchors': concept_anchors,
            'pca': self.pca,
            'umap': self.umap_reducer,
            'token_positions': token_positions,
            'layer_indices': layer_indices,
            'umap_embedding': X_umap
        }
        
        return result
    
    def llm_aided_concept_labeling(
        self,
        cluster_result: Dict[str, Any],
        prompt: str,
        generated_text: str,
        max_concepts: int = 10
    ) -> Dict[str, Any]:
        """
        Use an external LLM to label discovered clusters as human-interpretable concepts.
        
        Args:
            cluster_result: Result from discover_concepts_unsupervised
            prompt: Original prompt to the LLM
            generated_text: Generated text from the LLM
            max_concepts: Maximum number of concepts to label
            
        Returns:
            Updated cluster result with concept labels
        """
        if not self.use_openai_for_labeling:
            logger.warning("OpenAI labeling is disabled. Returning clusters without labels.")
            return cluster_result
        
        logger.info(f"Using {self.openai_model} to label discovered clusters as concepts")
        
        # Extract clusters from the result
        clusters = cluster_result['clusters']
        
        # Limit to max_concepts largest clusters
        cluster_sizes = {label: data['size'] for label, data in clusters.items()}
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        top_clusters = sorted_clusters[:max_concepts]
        
        # For each cluster, collect representative information
        labeled_clusters = {}
        for cluster_label, _ in top_clusters:
            cluster = clusters[cluster_label]
            
            # Get position range for the cluster
            position_range = cluster['position_range']
            
            # Extract the corresponding segment from the generated text
            # This is a simplification - in practice, you'd need to map positions to tokens to text
            # For now, we'll use a heuristic based on position ranges
            text_tokens = generated_text.split()
            segment_start = min(position_range[0], len(text_tokens) - 1)
            segment_end = min(position_range[1] + 1, len(text_tokens))
            segment_text = ' '.join(text_tokens[segment_start:segment_end])
            
            # Prepare a prompt for the LLM to label this cluster
            labeling_prompt = f"""
            You are helping to identify conceptual reasoning steps in an LLM's generation process.
            
            Original prompt: {prompt}
            
            Generated text segment: "{segment_text}"
            
            This segment appears around position {position_range} in the generation sequence.
            
            Based on this text segment, what single high-level concept or reasoning step does it most likely represent?
            Format your answer as a short phrase (2-5 words) with the specific concept name only, without any prefix or explanations.
            
            Examples of good concept names:
            - "Identifying variables"
            - "Retrieving relevant formula"
            - "Making causal inference"
            - "Evaluating evidence"
            - "Drawing final conclusion"
            """
            
            try:
                # Call OpenAI API to get concept label
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": labeling_prompt}],
                    temperature=0.2,
                    max_tokens=20
                )
                
                # Extract concept label from response
                concept_label = response.choices[0].message.content.strip()
                logger.info(f"Labeled cluster {cluster_label} as '{concept_label}'")
                
                # Update cluster with label
                cluster['concept_label'] = concept_label
                labeled_clusters[cluster_label] = cluster
                
            except Exception as e:
                logger.error(f"Error labeling cluster {cluster_label}: {str(e)}")
                # Keep the cluster without a label
                cluster['concept_label'] = f"Cluster {cluster_label}"
                labeled_clusters[cluster_label] = cluster
        
        # Update the cluster result with labeled clusters
        cluster_result['labeled_clusters'] = labeled_clusters
        
        return cluster_result
    
    def _visualize_clusters_2d(
        self,
        X_umap: np.ndarray,
        cluster_labels: np.ndarray,
        clusters: Dict[int, Dict[str, Any]],
        save_path: Optional[str] = None
    ):
        """
        Visualize clusters in 2D UMAP space.
        
        Args:
            X_umap: UMAP embedding of shape (n_samples, 2)
            cluster_labels: Cluster labels for each sample
            clusters: Dictionary of cluster information
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Get unique cluster labels
        unique_labels = np.unique(cluster_labels)
        
        # Use a colormap that distinguishes clusters well
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            plt.scatter(
                X_umap[mask, 0],
                X_umap[mask, 1],
                s=50,
                color=colors[i],
                label=f"Cluster {label} (n={clusters[label]['size']})",
                alpha=0.7
            )
            
            # Mark cluster center
            center = np.mean(X_umap[mask], axis=0)
            plt.scatter(
                center[0],
                center[1],
                s=200,
                color=colors[i],
                edgecolors='black',
                marker='*'
            )
            
            # Annotate with cluster number
            plt.annotate(
                f"{label}",
                xy=(center[0], center[1]),
                xytext=(center[0] + 0.2, center[1] + 0.2),
                fontsize=12,
                weight='bold'
            )
        
        plt.title("Concept Clusters (UMAP 2D)", fontsize=16)
        plt.xlabel("UMAP Dimension 1", fontsize=14)
        plt.ylabel("UMAP Dimension 2", fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster visualization to {save_path}")
        
        plt.close()
    
    def _visualize_clusters_3d(
        self,
        X_umap: np.ndarray,
        cluster_labels: np.ndarray,
        clusters: Dict[int, Dict[str, Any]],
        save_path: Optional[str] = None
    ):
        """
        Visualize clusters in 3D UMAP space.
        
        Args:
            X_umap: UMAP embedding of shape (n_samples, 3)
            cluster_labels: Cluster labels for each sample
            clusters: Dictionary of cluster information
            save_path: Path to save the visualization
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get unique cluster labels
        unique_labels = np.unique(cluster_labels)
        
        # Use a colormap that distinguishes clusters well
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            ax.scatter(
                X_umap[mask, 0],
                X_umap[mask, 1],
                X_umap[mask, 2],
                s=50,
                color=colors[i],
                label=f"Cluster {label} (n={clusters[label]['size']})",
                alpha=0.7
            )
            
            # Mark cluster center
            center = np.mean(X_umap[mask], axis=0)
            ax.scatter(
                center[0],
                center[1],
                center[2],
                s=200,
                color=colors[i],
                edgecolors='black',
                marker='*'
            )
            
            # Annotate with cluster number
            ax.text(
                center[0], 
                center[1], 
                center[2], 
                f"{label}",
                fontsize=12,
                weight='bold'
            )
        
        ax.set_title("Concept Clusters (UMAP 3D)", fontsize=16)
        ax.set_xlabel("UMAP Dimension 1", fontsize=14)
        ax.set_ylabel("UMAP Dimension 2", fontsize=14)
        ax.set_zlabel("UMAP Dimension 3", fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 3D cluster visualization to {save_path}")
        
        plt.close()
    
    def _visualize_concept_anchors(
        self,
        X_umap: np.ndarray,
        concept_anchors: Dict[str, Dict[str, Any]],
        token_positions: List[int],
        save_path: Optional[str] = None
    ):
        """
        Visualize concept anchors in 2D UMAP space.
        
        Args:
            X_umap: UMAP embedding of shape (n_samples, 2)
            concept_anchors: Dictionary of concept anchor information
            token_positions: Token positions for each sample
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 8))
        
        # Create a colormap for the concepts
        colors = plt.cm.tab20(np.linspace(0, 1, len(concept_anchors)))
        
        # Plot each concept
        for i, (concept, anchor) in enumerate(concept_anchors.items()):
            indices = anchor['indices']
            
            # Skip if no points
            if len(indices) == 0:
                continue
            
            # Get UMAP points for this concept
            concept_umap = anchor['umap_points']
            
            plt.scatter(
                concept_umap[:, 0],
                concept_umap[:, 1],
                s=50,
                color=colors[i],
                label=f"{concept} (n={len(indices)})",
                alpha=0.7
            )
            
            # Mark concept center
            centroid_umap = np.mean(concept_umap, axis=0)
            plt.scatter(
                centroid_umap[0],
                centroid_umap[1],
                s=200,
                color=colors[i],
                edgecolors='black',
                marker='*'
            )
            
            # Annotate with concept name
            plt.annotate(
                concept,
                xy=(centroid_umap[0], centroid_umap[1]),
                xytext=(centroid_umap[0] + 0.2, centroid_umap[1] + 0.2),
                fontsize=12,
                weight='bold'
            )
        
        plt.title("Concept Anchors (UMAP 2D)", fontsize=16)
        plt.xlabel("UMAP Dimension 1", fontsize=14)
        plt.ylabel("UMAP Dimension 2", fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved concept anchor visualization to {save_path}")
        
        plt.close()
        
        # Create a second visualization showing concept progression over token position
        plt.figure(figsize=(14, 8))
        
        # Determine y-coordinates based on concept index
        y_coords = np.linspace(0, 1, len(concept_anchors))
        concept_y = {concept: y for concept, y in zip(concept_anchors.keys(), y_coords)}
        
        # Plot the concept ranges
        for concept, anchor in concept_anchors.items():
            pos_range = anchor['position_range']
            plt.plot(
                [pos_range[0], pos_range[1]],
                [concept_y[concept], concept_y[concept]],
                linewidth=10,
                alpha=0.7,
                label=concept
            )
            
            # Add concept label
            plt.text(
                pos_range[0],
                concept_y[concept] + 0.02,
                concept,
                fontsize=10,
                verticalalignment='bottom'
            )
        
        plt.title("Concept Progression by Token Position", fontsize=16)
        plt.xlabel("Token Position", fontsize=14)
        plt.yticks([], [])  # Hide y-axis ticks
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            progression_path = save_path.replace('.png', '_progression.png')
            plt.savefig(progression_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved concept progression visualization to {progression_path}")
        
        plt.close()