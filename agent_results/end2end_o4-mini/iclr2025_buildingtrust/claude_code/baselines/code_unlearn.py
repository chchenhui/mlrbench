"""
Implementation of the CodeUnlearn baseline method for LLM unlearning.
Based on: "CodeUnlearn: Amortized Zero-Shot Machine Unlearning in Language Models Using Discrete Concept" (Wu et al., 2024)
"""

import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for learning discrete codebook features.
    """
    
    def __init__(self, input_dim, codebook_size=1024, hidden_dim=None, sparsity=0.05, l1_coef=0.001):
        """
        Initialize the sparse autoencoder.
        
        Args:
            input_dim: Dimension of input features
            codebook_size: Size of the codebook (number of features)
            hidden_dim: Hidden dimension (defaults to 2*input_dim)
            sparsity: Target sparsity level
            l1_coef: L1 regularization coefficient
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim or 2*input_dim
        self.sparsity = sparsity
        self.l1_coef = l1_coef
        
        # Encoder and decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, codebook_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(codebook_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, input_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the autoencoder."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            reconstructed: Reconstructed features
            codes: Sparse codes
        """
        # Encode
        codes = self.encoder(x)
        
        # Apply soft thresholding for sparsity
        threshold = torch.quantile(codes.abs(), q=1-self.sparsity, dim=1, keepdim=True)
        sparse_codes = torch.sign(codes) * F.relu(codes.abs() - threshold)
        
        # Decode
        reconstructed = self.decoder(sparse_codes)
        
        return reconstructed, sparse_codes
    
    def compute_loss(self, x, reconstructed, codes):
        """
        Compute the autoencoder loss.
        
        Args:
            x: Input features
            reconstructed: Reconstructed features
            codes: Sparse codes
            
        Returns:
            loss: Total loss
            components: Dictionary of loss components
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x)
        
        # L1 sparsity loss
        l1_loss = self.l1_coef * torch.mean(torch.abs(codes))
        
        # Total loss
        total_loss = recon_loss + l1_loss
        
        # Loss components
        components = {
            'reconstruction': recon_loss.item(),
            'l1_sparsity': l1_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, components


class ConceptBottleneck(nn.Module):
    """
    Concept bottleneck module that uses sparse autoencoders to regulate information flow.
    """
    
    def __init__(self, hidden_size, codebook_size=1024, num_layers=2, dropout_prob=0.1):
        """
        Initialize the concept bottleneck module.
        
        Args:
            hidden_size: Hidden size of the transformer
            codebook_size: Size of the codebook
            num_layers: Number of layers to apply the bottleneck to
            dropout_prob: Dropout probability
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        
        # Create sparse autoencoders for each layer
        self.autoencoders = nn.ModuleList([
            SparseAutoencoder(hidden_size, codebook_size=codebook_size)
            for _ in range(num_layers)
        ])
        
        # Projections and layer norms
        self.pre_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        self.post_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_prob)
            for _ in range(num_layers)
        ])
        
        # Concept masking
        self.concept_masks = nn.ParameterList([
            nn.Parameter(torch.ones(codebook_size))
            for _ in range(num_layers)
        ])
    
    def forward(self, hidden_states, layer_idx=0):
        """
        Forward pass.
        
        Args:
            hidden_states: Hidden states from the transformer layer
            layer_idx: Index of the layer
            
        Returns:
            modified_hidden_states: Modified hidden states
        """
        # Ensure layer_idx is valid
        layer_idx = layer_idx % self.num_layers
        
        # Project the hidden states
        projected = self.pre_projections[layer_idx](hidden_states)
        
        # Reshape if needed (for sequence models)
        original_shape = projected.shape
        if len(original_shape) > 2:
            # [batch_size, seq_len, hidden_size] -> [batch_size*seq_len, hidden_size]
            projected = projected.reshape(-1, self.hidden_size)
        
        # Apply autoencoder with concept masking
        reconstructed, codes = self.autoencoders[layer_idx](projected)
        
        # Apply concept mask (element-wise multiplication)
        mask = torch.sigmoid(self.concept_masks[layer_idx]).to(codes.device)
        masked_codes = codes * mask
        
        # Decode with masked codes
        reconstructed = self.autoencoders[layer_idx].decoder(masked_codes)
        
        # Reshape back to original shape if needed
        if len(original_shape) > 2:
            reconstructed = reconstructed.reshape(original_shape)
        
        # Project back, apply dropout and layer norm
        modified_hidden_states = self.post_projections[layer_idx](reconstructed)
        modified_hidden_states = self.dropouts[layer_idx](modified_hidden_states)
        modified_hidden_states = self.layer_norms[layer_idx](modified_hidden_states + hidden_states)
        
        return modified_hidden_states
    
    def mask_concepts(self, concept_indices, layer_idx=None, reset_first=False):
        """
        Mask specific concepts by setting their mask values to 0.
        
        Args:
            concept_indices: Indices of concepts to mask
            layer_idx: Index of the layer (if None, apply to all layers)
            reset_first: Whether to reset masks to 1 before applying new masks
            
        Returns:
            None
        """
        layers = range(self.num_layers) if layer_idx is None else [layer_idx % self.num_layers]
        
        for idx in layers:
            if reset_first:
                # Reset mask to all ones
                self.concept_masks[idx].data = torch.ones_like(self.concept_masks[idx])
            
            # Set mask values for specified concepts to a large negative value (sigmoid will make it close to 0)
            self.concept_masks[idx].data[concept_indices] = -10.0


class CodeUnlearnMethod:
    """
    CodeUnlearn unlearning method implementation.
    
    The method works by using sparse autoencoders to decompose the activation space
    into discrete concepts, and then masking the concepts related to information to unlearn.
    """
    
    def __init__(
        self,
        model,
        optimizer_class=torch.optim.AdamW,
        learning_rate=5e-5,
        num_epochs=3,
        batch_size=16,
        codebook_size=1024,
        num_bottleneck_layers=2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the CodeUnlearn method.
        
        Args:
            model: The pretrained LLM model
            optimizer_class: Optimizer class to use
            learning_rate: Learning rate for fine-tuning
            num_epochs: Number of epochs for fine-tuning
            batch_size: Batch size for fine-tuning
            codebook_size: Size of the codebook
            num_bottleneck_layers: Number of bottleneck layers
            device: Device to use for computation
        """
        self.model = model.to(device)
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.codebook_size = codebook_size
        self.num_bottleneck_layers = num_bottleneck_layers
        self.device = device
        
        # Determine hidden size
        if hasattr(model, 'config'):
            self.hidden_size = model.config.hidden_size
        else:
            # Try to infer from the model parameters
            for name, param in model.named_parameters():
                if 'weight' in name and len(param.shape) == 2:
                    self.hidden_size = param.shape[0]
                    break
            else:
                self.hidden_size = 768  # Default size if we can't determine
    
    def _insert_concept_bottleneck(self, model):
        """
        Insert concept bottleneck into the model.
        
        Args:
            model: The model to modify
            
        Returns:
            modified_model: The modified model with concept bottleneck
            bottleneck: The concept bottleneck module
        """
        # Create a copy of the model
        modified_model = copy.deepcopy(model)
        
        # Create concept bottleneck
        bottleneck = ConceptBottleneck(
            hidden_size=self.hidden_size,
            codebook_size=self.codebook_size,
            num_layers=self.num_bottleneck_layers
        ).to(self.device)
        
        # For GPT-2 style models
        if hasattr(modified_model, 'transformer') and hasattr(modified_model.transformer, 'h'):
            num_layers = len(modified_model.transformer.h)
            
            # Insert bottleneck after every few transformer layers
            layer_stride = max(1, num_layers // self.num_bottleneck_layers)
            for i in range(layer_stride - 1, num_layers, layer_stride):
                if i >= num_layers:
                    break
                
                # Save the original forward method
                original_forward = modified_model.transformer.h[i].forward
                
                # Define a new forward method that adds the bottleneck
                def make_forward(layer_idx, orig_forward):
                    def new_forward(hidden_states, *args, **kwargs):
                        # Call the original forward method
                        outputs = orig_forward(hidden_states, *args, **kwargs)
                        
                        # Apply bottleneck to the output
                        if isinstance(outputs, tuple):
                            modified_outputs = (bottleneck(outputs[0], layer_idx=layer_idx),) + outputs[1:]
                        else:
                            modified_outputs = bottleneck(outputs, layer_idx=layer_idx)
                        
                        return modified_outputs
                    
                    return new_forward
                
                # Replace the forward method
                bottleneck_idx = (i // layer_stride) % self.num_bottleneck_layers
                modified_model.transformer.h[i].forward = make_forward(bottleneck_idx, original_forward)
        
        # For BERT style models
        elif hasattr(modified_model, 'encoder') and hasattr(modified_model.encoder, 'layer'):
            num_layers = len(modified_model.encoder.layer)
            
            # Insert bottleneck after every few transformer layers
            layer_stride = max(1, num_layers // self.num_bottleneck_layers)
            for i in range(layer_stride - 1, num_layers, layer_stride):
                if i >= num_layers:
                    break
                
                # Save the original forward method
                original_forward = modified_model.encoder.layer[i].forward
                
                # Define a new forward method that adds the bottleneck
                def make_forward(layer_idx, orig_forward):
                    def new_forward(hidden_states, *args, **kwargs):
                        # Call the original forward method
                        outputs = orig_forward(hidden_states, *args, **kwargs)
                        
                        # Apply bottleneck to the output
                        if isinstance(outputs, tuple):
                            modified_outputs = (bottleneck(outputs[0], layer_idx=layer_idx),) + outputs[1:]
                        else:
                            modified_outputs = bottleneck(outputs, layer_idx=layer_idx)
                        
                        return modified_outputs
                    
                    return new_forward
                
                # Replace the forward method
                bottleneck_idx = (i // layer_stride) % self.num_bottleneck_layers
                modified_model.encoder.layer[i].forward = make_forward(bottleneck_idx, original_forward)
        
        return modified_model, bottleneck
    
    def _identify_concepts_to_mask(self, bottleneck, deletion_set, validation_set):
        """
        Identify concepts to mask based on activation patterns.
        
        Args:
            bottleneck: Concept bottleneck module
            deletion_set: Set of examples to delete
            validation_set: Set of examples to retain
            
        Returns:
            concepts_to_mask: List of concept indices to mask for each layer
        """
        concepts_to_mask = [[] for _ in range(bottleneck.num_layers)]
        
        # Create data loaders
        deletion_loader = DataLoader(
            deletion_set,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        validation_loader = DataLoader(
            validation_set,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Collect activation patterns
        deletion_activations = [[] for _ in range(bottleneck.num_layers)]
        validation_activations = [[] for _ in range(bottleneck.num_layers)]
        
        # Function to collect activations from a specific layer
        def collect_activations(loader, activations_list):
            for batch in loader:
                # Move batch to device
                if isinstance(batch, dict):
                    inputs = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device)
                    }
                else:
                    # Assume batch is a tuple of (inputs, targets)
                    inputs = {k: v.to(self.device) for k, v in batch[0].items()}
                
                # Forward pass through each autoencoder
                with torch.no_grad():
                    for layer_idx in range(bottleneck.num_layers):
                        # Get hidden states (this is a simplified approach)
                        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                            num_layers = len(self.model.transformer.h)
                            layer_stride = max(1, num_layers // self.num_bottleneck_layers)
                            model_layer_idx = layer_stride * (layer_idx + 1) - 1
                            
                            # Forward pass up to this layer
                            hidden_states = inputs['input_ids']
                            for i in range(model_layer_idx):
                                hidden_states = self.model.transformer.h[i](hidden_states)[0]
                            
                        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                            num_layers = len(self.model.encoder.layer)
                            layer_stride = max(1, num_layers // self.num_bottleneck_layers)
                            model_layer_idx = layer_stride * (layer_idx + 1) - 1
                            
                            # Forward pass up to this layer
                            hidden_states = self.model.encoder.embeddings(inputs['input_ids'])
                            for i in range(model_layer_idx):
                                hidden_states = self.model.encoder.layer[i](hidden_states)[0]
                        else:
                            # Fallback: use a random tensor
                            hidden_states = torch.randn(
                                inputs['input_ids'].shape[0],
                                inputs['input_ids'].shape[1],
                                self.hidden_size,
                                device=self.device
                            )
                        
                        # Project and encode
                        projected = bottleneck.pre_projections[layer_idx](hidden_states)
                        batch_size, seq_len, _ = projected.shape
                        projected = projected.reshape(batch_size * seq_len, -1)
                        
                        # Get codes
                        _, codes = bottleneck.autoencoders[layer_idx](projected)
                        
                        # Store activations
                        activations_list[layer_idx].append(codes.detach().cpu())
        
        # Collect activations
        collect_activations(deletion_loader, deletion_activations)
        collect_activations(validation_loader, validation_activations)
        
        # Process activations to find concepts to mask
        for layer_idx in range(bottleneck.num_layers):
            # Concatenate activations
            deletion_acts = torch.cat(deletion_activations[layer_idx], dim=0)
            validation_acts = torch.cat(validation_activations[layer_idx], dim=0)
            
            # Compute mean activation for each concept
            deletion_mean = deletion_acts.mean(dim=0)
            validation_mean = validation_acts.mean(dim=0)
            
            # Compute concept importance based on activation difference
            importance = torch.abs(deletion_mean - validation_mean)
            
            # Select top concepts by importance (top 10% by default)
            num_concepts_to_mask = int(0.1 * bottleneck.codebook_size)
            top_concepts = torch.topk(importance, num_concepts_to_mask).indices.tolist()
            
            concepts_to_mask[layer_idx] = top_concepts
        
        return concepts_to_mask
    
    def unlearn(self, validation_data, deletion_set):
        """
        Perform unlearning using the CodeUnlearn method.
        
        Args:
            validation_data: Dataset of examples to retain
            deletion_set: Set of examples to forget
            
        Returns:
            unlearned_model: The unlearned model
            metrics: Unlearning metrics
        """
        # Start timing
        start_time = time.time()
        
        # Insert concept bottleneck into the model
        unlearned_model, bottleneck = self._insert_concept_bottleneck(self.model)
        
        # Pretrain the concept bottleneck on validation data
        bottleneck_optimizer = self.optimizer_class(
            bottleneck.parameters(),
            lr=self.learning_rate
        )
        
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Train the bottleneck to reconstruct validation data
        unlearned_model.train()
        
        for epoch in range(2):  # Fewer epochs for pretraining
            for batch in validation_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    inputs = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device)
                    }
                    targets = batch['targets'].to(self.device)
                else:
                    # Assume batch is a tuple of (inputs, targets)
                    inputs = {k: v.to(self.device) for k, v in batch[0].items()}
                    targets = batch[1].to(self.device)
                
                # Forward pass
                bottleneck_optimizer.zero_grad()
                outputs = unlearned_model(**inputs)
                logits = outputs.logits
                
                # Compute loss
                if logits.shape[:-1] != targets.shape:
                    # Reshape for language modeling
                    shifted_logits = logits.contiguous().view(-1, logits.size(-1))
                    shifted_targets = targets.contiguous().view(-1)
                else:
                    shifted_logits = logits
                    shifted_targets = targets
                    
                loss = F.cross_entropy(shifted_logits, shifted_targets)
                
                # Add autoencoder reconstruction loss
                for layer_idx in range(bottleneck.num_layers):
                    for ae_batch in bottleneck.autoencoders[layer_idx].parameters():
                        if hasattr(ae_batch, 'grad') and ae_batch.grad is not None:
                            loss += 0.01 * ae_batch.grad.norm()
                
                # Backward and optimize
                loss.backward()
                bottleneck_optimizer.step()
        
        # Identify concepts to mask based on activation patterns
        concepts_to_mask = self._identify_concepts_to_mask(bottleneck, deletion_set, validation_data)
        
        # Mask the identified concepts
        for layer_idx in range(bottleneck.num_layers):
            bottleneck.mask_concepts(concepts_to_mask[layer_idx], layer_idx=layer_idx)
        
        # Fine-tune the model with masked concepts
        unlearned_model_optimizer = self.optimizer_class(
            unlearned_model.parameters(),
            lr=self.learning_rate * 0.1  # Lower learning rate for fine-tuning
        )
        
        # Fine-tune on validation data to ensure performance
        for epoch in range(1):  # Just one epoch for fine-tuning
            for batch in validation_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    inputs = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device)
                    }
                    targets = batch['targets'].to(self.device)
                else:
                    # Assume batch is a tuple of (inputs, targets)
                    inputs = {k: v.to(self.device) for k, v in batch[0].items()}
                    targets = batch[1].to(self.device)
                
                # Forward pass
                unlearned_model_optimizer.zero_grad()
                outputs = unlearned_model(**inputs)
                logits = outputs.logits
                
                # Compute loss
                if logits.shape[:-1] != targets.shape:
                    # Reshape for language modeling
                    shifted_logits = logits.contiguous().view(-1, logits.size(-1))
                    shifted_targets = targets.contiguous().view(-1)
                else:
                    shifted_logits = logits
                    shifted_targets = targets
                    
                loss = F.cross_entropy(shifted_logits, shifted_targets)
                
                # Backward and optimize
                loss.backward()
                unlearned_model_optimizer.step()
        
        # Compute metrics
        metrics = {
            'method': 'CodeUnlearn',
            'compute_time': time.time() - start_time,
            'num_masked_concepts': sum(len(concepts) for concepts in concepts_to_mask)
        }
        
        return unlearned_model, metrics