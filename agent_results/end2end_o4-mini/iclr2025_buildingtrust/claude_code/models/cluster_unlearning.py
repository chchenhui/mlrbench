"""
Main implementation of the Cluster-Driven Certified Unlearning framework.
"""

import os
import copy
import time
import torch
import numpy as np
from .spectral_clustering import RepresentationClustering
from .influence_scores import InfluenceScoreApproximation
from .gradient_surgery import TargetedLowRankGradientSurgery
from .fisher_certification import FisherInformationCertification


class ClusterDrivenCertifiedUnlearning:
    """
    Main class for the Cluster-Driven Certified Unlearning framework.
    """
    
    def __init__(
        self,
        model,
        n_clusters=10,
        embedding_dim=64,
        influence_threshold=0.1,
        learning_rates=None,
        epsilon=0.1,
        adapter_rank=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the unlearning framework.
        
        Args:
            model (torch.nn.Module): The pretrained LLM model
            n_clusters (int): Number of clusters to form
            embedding_dim (int): Dimension of the spectral embedding
            influence_threshold (float): Threshold for influence scores
            learning_rates (dict, optional): Dictionary mapping cluster indices to learning rates
            epsilon (float): KL divergence threshold for certification
            adapter_rank (int): Rank of the low-rank adapters for sequential deletions
            device (str): Device to use for computation
        """
        self.model = model.to(device)
        self.device = device
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        
        # Initialize components
        self.clustering = RepresentationClustering(
            n_clusters=n_clusters,
            embedding_dim=embedding_dim
        )
        
        self.influence_scorer = InfluenceScoreApproximation(
            model=model,
            threshold=influence_threshold
        )
        
        self.gradient_surgery = TargetedLowRankGradientSurgery(
            model=model,
            learning_rates=learning_rates,
            adapter_rank=adapter_rank
        )
        
        # Store cluster bases and activations
        self.cluster_bases = {}
        self.all_activations = None
        self.layer_idx = None
        
        # Store sequential deletion info
        self.deletion_history = []
        self.current_unlearning_session = None
        
    def extract_activations(self, data_loader, layer_idx=-1):
        """
        Extract hidden-layer activations from the model.
        
        Args:
            data_loader: DataLoader providing samples for activation extraction
            layer_idx (int): Index of the layer to extract activations from (-1 for final layer)
            
        Returns:
            activations (torch.Tensor): Extracted activations
        """
        self.model.eval()
        
        # Store layer index
        self.layer_idx = layer_idx
        
        # Initialize hook for extracting activations
        activations = []
        
        def hook_fn(module, input, output):
            # For transformer-based models, output may be a tuple
            if isinstance(output, tuple):
                # Usually the first element is the hidden states
                hidden_states = output[0]
            else:
                hidden_states = output
                
            # Store the activations
            activations.append(hidden_states.detach())
        
        # Register forward hook on the specified layer
        if hasattr(self.model, 'transformer'):
            # GPT-2 style models
            if layer_idx < 0:
                layer = list(self.model.transformer.h)[-1]
            else:
                layer = self.model.transformer.h[layer_idx]
        elif hasattr(self.model, 'encoder'):
            # BERT style models
            if layer_idx < 0:
                layer = list(self.model.encoder.layer)[-1]
            else:
                layer = self.model.encoder.layer[layer_idx]
        else:
            # Try a more generic approach
            layers = [m for m in self.model.modules() if len(list(m.children())) == 0]
            if layer_idx < 0:
                layer = layers[-1]
            else:
                layer = layers[layer_idx]
        
        handle = layer.register_forward_hook(hook_fn)
        
        # Run data through the model to collect activations
        for batch in data_loader:
            inputs = batch[0]  # Assuming batch is (inputs, targets)
            self.model(**inputs)
            
        # Remove the hook
        handle.remove()
        
        # Concatenate activations
        all_activations = torch.cat(activations, dim=0)
        
        # Reshape to 2D if needed
        if len(all_activations.shape) > 2:
            # For sequence models, average over the sequence dimension
            all_activations = all_activations.mean(dim=1)
            
        # Store for later use
        self.all_activations = all_activations
        
        return all_activations
        
    def cluster_representations(self, activations=None):
        """
        Cluster the model's representations using hierarchical spectral clustering.
        
        Args:
            activations (torch.Tensor, optional): Hidden-layer activations
            
        Returns:
            cluster_assignments (np.ndarray): Cluster assignments for each sample
        """
        # Use provided activations or previously extracted ones
        if activations is None:
            if self.all_activations is None:
                raise ValueError("No activations available. Call extract_activations() first.")
            activations = self.all_activations
            
        # Perform clustering
        self.clustering.fit(activations)
        
        # Get cluster assignments
        cluster_assignments = self.clustering.clusters
        
        # Compute cluster bases
        for k in range(self.n_clusters):
            cluster_mask = cluster_assignments == k
            if np.sum(cluster_mask) > 0:
                self.cluster_bases[k] = self.clustering.get_cluster_basis(activations, k)
                
        return cluster_assignments
        
    def unlearn(self, deletion_set, validation_data=None, test_data=None):
        """
        Perform unlearning for a set of examples to be deleted.
        
        Args:
            deletion_set (list): List of (inputs, targets) tuples for data to be deleted
            validation_data (list, optional): Validation data for computing influence scores
            test_data (DataLoader, optional): Test data for certification
            
        Returns:
            unlearned_model (torch.nn.Module): The unlearned model
            certificate (dict): Certification details
            metrics (dict): Unlearning metrics
        """
        # Start timing
        start_time = time.time()
        
        # Start a new unlearning session
        session_id = len(self.deletion_history) + 1
        self.current_unlearning_session = {
            'session_id': session_id,
            'deletion_set_size': len(deletion_set),
            'start_time': start_time
        }
        
        # 1. Identify affected clusters using influence scores
        influence_scores = self.influence_scorer.compute_cluster_influence(
            deletion_set, self.cluster_bases, validation_data
        )
        
        affected_clusters = self.influence_scorer.identify_affected_clusters(influence_scores)
        
        # 2. Apply targeted low-rank gradient surgery
        # Create a copy of the model for unlearning
        unlearned_model = copy.deepcopy(self.model)
        
        # Initialize learning rates if not provided
        if not hasattr(self.gradient_surgery, 'learning_rates') or not self.gradient_surgery.learning_rates:
            # Set learning rates proportional to influence scores
            learning_rates = {}
            max_score = max(abs(s) for s in influence_scores.values())
            for cluster_idx, score in influence_scores.items():
                # Normalize and scale (0.005 - 0.05 range)
                learning_rates[cluster_idx] = 0.005 + 0.045 * abs(score) / max_score
                
            # Update gradient surgery module
            self.gradient_surgery.learning_rates = learning_rates
            
        # Apply surgery to each affected cluster
        for cluster_idx in affected_clusters:
            # Compute deletion gradient
            deletion_grad = self.gradient_surgery.compute_deletion_gradient(deletion_set)
            
            # Project gradient onto cluster subspace
            projected_grad, _ = self.gradient_surgery.project_gradient(
                deletion_grad, self.cluster_bases[cluster_idx]
            )
            
            # Apply gradient update
            updated_params = self.gradient_surgery.apply_gradient_update(
                cluster_idx, projected_grad
            )
            
            # Create low-rank adapter for sequential deletions
            self.gradient_surgery.create_low_rank_adapter(
                cluster_idx, deletion_set, self.cluster_bases[cluster_idx]
            )
            
            # Update model parameters
            with torch.no_grad():
                for i, (param, updated) in enumerate(zip(unlearned_model.parameters(), updated_params)):
                    param.copy_(updated)
        
        # 3. Certify the unlearning operation
        if test_data is not None:
            certifier = FisherInformationCertification(
                original_model=self.model,
                unlearned_model=unlearned_model,
                epsilon=self.epsilon
            )
            
            is_certified, kl_div, certificate = certifier.certify(test_data)
            
            # Refine learning rates and reapply if not certified
            if not is_certified:
                # Decrease learning rates and retry
                for cluster_idx in affected_clusters:
                    self.gradient_surgery.learning_rates[cluster_idx] *= 0.5
                    
                # Retry unlearning
                return self.unlearn(deletion_set, validation_data, test_data)
        else:
            is_certified = None
            kl_div = None
            certificate = None
            
        # Compute metrics
        metrics = {
            'affected_clusters': affected_clusters,
            'influence_scores': influence_scores,
            'compute_time': time.time() - start_time,
            'kl_divergence': kl_div,
            'is_certified': is_certified
        }
        
        # Update session info
        self.current_unlearning_session.update({
            'affected_clusters': affected_clusters,
            'is_certified': is_certified,
            'kl_divergence': kl_div,
            'compute_time': time.time() - start_time
        })
        
        # Add to deletion history
        self.deletion_history.append(self.current_unlearning_session)
        
        return unlearned_model, certificate, metrics
        
    def sequential_unlearn(self, new_deletion_set, validation_data=None, test_data=None):
        """
        Apply unlearning for a new deletion set while preserving previous deletions.
        
        Args:
            new_deletion_set (list): List of (inputs, targets) tuples for new data to be deleted
            validation_data (list, optional): Validation data for computing influence scores
            test_data (DataLoader, optional): Test data for certification
            
        Returns:
            unlearned_model (torch.nn.Module): The updated unlearned model
            certificate (dict): Certification details
            metrics (dict): Unlearning metrics
        """
        # Start timing
        start_time = time.time()
        
        # Start a new unlearning session
        session_id = len(self.deletion_history) + 1
        self.current_unlearning_session = {
            'session_id': session_id,
            'deletion_set_size': len(new_deletion_set),
            'start_time': start_time,
            'is_sequential': True
        }
        
        # 1. Identify affected clusters for the new deletion set
        influence_scores = self.influence_scorer.compute_cluster_influence(
            new_deletion_set, self.cluster_bases, validation_data
        )
        
        affected_clusters = self.influence_scorer.identify_affected_clusters(influence_scores)
        
        # 2. Create and apply low-rank adapters for the new deletion set
        for cluster_idx in affected_clusters:
            # Create low-rank adapter
            self.gradient_surgery.create_low_rank_adapter(
                cluster_idx, new_deletion_set, self.cluster_bases[cluster_idx]
            )
            
        # Apply all accumulated adapters to get the updated model
        unlearned_model = copy.deepcopy(self.model)
        
        # Transfer adapters to new gradient surgery instance
        new_gradient_surgery = TargetedLowRankGradientSurgery(
            model=unlearned_model,
            learning_rates=self.gradient_surgery.learning_rates,
            adapter_rank=self.gradient_surgery.adapter_rank
        )
        new_gradient_surgery.adapters = copy.deepcopy(self.gradient_surgery.adapters)
        
        # Apply adapters
        new_gradient_surgery.update_model_with_adapters()
        
        # 3. Certify the unlearning operation
        if test_data is not None:
            certifier = FisherInformationCertification(
                original_model=self.model,
                unlearned_model=unlearned_model,
                epsilon=self.epsilon
            )
            
            is_certified, kl_div, certificate = certifier.certify(test_data)
            
            # Handle certification failure
            if not is_certified:
                # Reduce learning rates for the new adapters
                for cluster_idx in affected_clusters:
                    adapter_name = f"adapter_{cluster_idx}"
                    if adapter_name in new_gradient_surgery.adapters:
                        # Halve the learning rate for this adapter
                        new_gradient_surgery.adapters[adapter_name].learning_rate *= 0.5
                
                # Reapply adapters with reduced learning rates
                new_gradient_surgery.update_model_with_adapters()
                
                # Recertify
                is_certified, kl_div, certificate = certifier.certify(test_data)
        else:
            is_certified = None
            kl_div = None
            certificate = None
            
        # Compute metrics
        metrics = {
            'affected_clusters': affected_clusters,
            'influence_scores': influence_scores,
            'compute_time': time.time() - start_time,
            'kl_divergence': kl_div,
            'is_certified': is_certified
        }
        
        # Update session info
        self.current_unlearning_session.update({
            'affected_clusters': affected_clusters,
            'is_certified': is_certified,
            'kl_divergence': kl_div,
            'compute_time': time.time() - start_time
        })
        
        # Add to deletion history
        self.deletion_history.append(self.current_unlearning_session)
        
        # Update gradient surgery with new adapters
        self.gradient_surgery = new_gradient_surgery
        
        return unlearned_model, certificate, metrics
        
    def save(self, save_dir):
        """
        Save the unlearning framework components.
        
        Args:
            save_dir (str): Directory to save to
            
        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save cluster bases
        np.save(os.path.join(save_dir, 'cluster_bases.npy'), self.cluster_bases)
        
        # Save cluster assignments
        np.save(os.path.join(save_dir, 'cluster_assignments.npy'), self.clustering.clusters)
        
        # Save learning rates
        np.save(os.path.join(save_dir, 'learning_rates.npy'), self.gradient_surgery.learning_rates)
        
        # Save deletion history
        np.save(os.path.join(save_dir, 'deletion_history.npy'), self.deletion_history)
        
        # Save adapters (this requires specialized serialization)
        adapters_state = {}
        for name, adapter in self.gradient_surgery.adapters.items():
            adapters_state[name] = adapter.state_dict()
            
        torch.save(adapters_state, os.path.join(save_dir, 'adapters.pth'))
        
    def load(self, load_dir, model):
        """
        Load the unlearning framework components.
        
        Args:
            load_dir (str): Directory to load from
            model (torch.nn.Module): The model to use
            
        Returns:
            self: The loaded framework
        """
        # Load cluster bases
        self.cluster_bases = np.load(os.path.join(load_dir, 'cluster_bases.npy'), allow_pickle=True).item()
        
        # Load cluster assignments
        self.clustering.clusters = np.load(os.path.join(load_dir, 'cluster_assignments.npy'), allow_pickle=True)
        
        # Load learning rates
        self.gradient_surgery.learning_rates = np.load(os.path.join(load_dir, 'learning_rates.npy'), allow_pickle=True).item()
        
        # Load deletion history
        self.deletion_history = np.load(os.path.join(load_dir, 'deletion_history.npy'), allow_pickle=True).tolist()
        
        # Load adapters (requires recreation of adapter instances)
        adapters_state = torch.load(os.path.join(load_dir, 'adapters.pth'))
        
        # Recreate and load each adapter
        for name, state in adapters_state.items():
            # Parse name to get cluster index
            cluster_idx = int(name.split('_')[1])
            
            # Create dummy adapter (will be populated with loaded state)
            adapter = self.gradient_surgery.create_low_rank_adapter(
                cluster_idx, [], self.cluster_bases[cluster_idx]
            )
            
            # Load state
            adapter.load_state_dict(state)
            
        # Update model
        self.model = model
        self.gradient_surgery.model = model
        self.influence_scorer.model = model
        
        return self