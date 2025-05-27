"""
Implementation of meta-learning components for MeLPA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Avoid using higher for simplicity
# import higher


class InitializationNetwork(nn.Module):
    """
    Network that generates initial adapter parameters based on task context.
    """
    def __init__(self, input_dim, adapter_shapes, hidden_dims=None, use_task_context=True):
        super().__init__()
        self.input_dim = input_dim
        self.adapter_shapes = adapter_shapes
        self.use_task_context = use_task_context

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Build network layers
        layers = []
        prev_dim = input_dim if use_task_context else 1  # If no context, use a dummy input

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        self.backbone = nn.Sequential(*layers)

        # Create output heads for each adapter parameter tensor
        self.output_heads = nn.ModuleDict()
        for name, shape in adapter_shapes.items():
            # Calculate total number of parameters in this tensor
            total_params = torch.prod(torch.tensor(shape)).item()
            # Replace dots in names with underscores to avoid ModuleDict key errors
            safe_name = name.replace(".", "_")
            self.output_heads[safe_name] = nn.Linear(prev_dim, total_params)
    
    def forward(self, task_context=None):
        """
        Generate initial adapter parameters.

        Args:
            task_context: Optional tensor encoding task context information.
                          If None and use_task_context is True, raises an error.
                          If use_task_context is False, ignores this argument.

        Returns:
            Dictionary mapping parameter names to parameter tensors.
        """
        if self.use_task_context and task_context is None:
            raise ValueError("Task context must be provided when use_task_context=True")

        # Use task context if provided, otherwise use a dummy input
        if self.use_task_context:
            x = task_context
        else:
            x = torch.ones(1, 1, device=next(self.parameters()).device)

        # Forward through backbone
        features = self.backbone(x)

        # Generate parameters for each adapter tensor
        adapter_params = {}
        for name, shape in self.adapter_shapes.items():
            # Use the safe name for the ModuleDict lookup
            safe_name = name.replace(".", "_")
            flat_params = self.output_heads[safe_name](features)
            adapter_params[name] = flat_params.view(*shape)

        return adapter_params


class UpdateMechanism(nn.Module):
    """
    Learned update mechanism for adapter parameters.
    This can range from simple learned learning rates to more sophisticated neural network
    update rules.
    """
    def __init__(self, method="learned_lr", **kwargs):
        super().__init__()
        self.method = method
        
        if method == "learned_lr":
            # Simple learned per-parameter learning rates
            self.param_names = kwargs.get("param_names", [])
            # Replace dots in names to avoid ParameterDict key errors
            self.param_names_map = {name: name.replace(".", "_") for name in self.param_names}
            self.learning_rates = nn.ParameterDict({
                safe_name: nn.Parameter(torch.ones(1) * 0.01)
                for name, safe_name in self.param_names_map.items()
            })
        
        elif method == "mlp_update":
            # MLP-based update rule
            input_dim = kwargs.get("input_dim", 2)  # grad and param values
            hidden_dims = kwargs.get("hidden_dims", [64, 32])
            
            layers = []
            prev_dim = input_dim
            for dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(nn.ReLU())
                prev_dim = dim
            
            layers.append(nn.Linear(prev_dim, 1))
            self.update_network = nn.Sequential(*layers)
        
        else:
            raise ValueError(f"Update method {method} not supported")
    
    def forward(self, parameters, gradients):
        """
        Compute parameter updates based on current parameters and gradients.
        
        Args:
            parameters: Dictionary of current parameter values
            gradients: Dictionary of current parameter gradients
        
        Returns:
            Dictionary of parameter updates
        """
        updates = {}
        
        if self.method == "learned_lr":
            for name in self.param_names:
                if name in parameters and name in gradients:
                    # Use the safe name for the ParameterDict lookup
                    safe_name = self.param_names_map.get(name, name.replace(".", "_"))
                    updates[name] = -self.learning_rates[safe_name] * gradients[name]
        
        elif self.method == "mlp_update":
            for name, param in parameters.items():
                if name in gradients:
                    # Stack parameter and gradient for each parameter element
                    batch_size = param.numel()
                    inputs = torch.stack([
                        param.flatten(),
                        gradients[name].flatten()
                    ], dim=1)  # [batch_size, 2]
                    
                    # Compute updates for each parameter element
                    update = self.update_network(inputs).view_as(param)
                    updates[name] = update
        
        return updates


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) for adapter-based meta-learning.
    This implementation is based on the original MAML algorithm but adapted
    for our MeLPA framework.
    """
    def __init__(self, model, adapter_name, inner_lr=0.01, first_order=False):
        super().__init__()
        self.model = model
        self.adapter_name = adapter_name
        self.inner_lr = inner_lr  # Inner loop learning rate
        self.first_order = first_order  # Whether to use first-order approximation
    
    def adapt(self, support_batch, inner_steps=5):
        """
        Adapt the model to the support set (inner loop optimization).
        
        Args:
            support_batch: Batch of support set data
            inner_steps: Number of inner optimization steps
        
        Returns:
            Adapted model
        """
        # Make a copy of the model for inner loop updates
        adapted_model = copy.deepcopy(self.model)
        
        # Get adapter parameters
        adapter_params = adapted_model.get_adapter_parameters(self.adapter_name)
        optimizer = torch.optim.SGD(adapter_params, lr=self.inner_lr)
        
        # Perform inner loop updates
        for _ in range(inner_steps):
            loss = self._compute_loss(adapted_model, support_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def _compute_loss(self, model, batch):
        """Compute loss for a batch of data."""
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        outputs = model(inputs, adapter_name=self.adapter_name)

        # Handle different output formats
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            raise ValueError("Model output format not supported")

        loss = F.cross_entropy(logits, batch['labels'])
        return loss
    
    def forward(self, support_batch, query_batch, inner_steps=5):
        """
        Forward pass for MAML.

        Args:
            support_batch: Batch of support set data
            query_batch: Batch of query set data
            inner_steps: Number of inner optimization steps

        Returns:
            Loss on the query set after adaptation
        """
        # Always use first-order approximation for simplicity
        # Adapt the model on the support set
        adapted_model = self.adapt(support_batch, inner_steps)

        # Compute loss on the query set with the adapted model
        query_loss = self._compute_loss(adapted_model, query_batch)
        return query_loss

        # Original implementation with higher:
        """
        if self.first_order:
            # First-order approximation (no second derivatives)
            # Adapt the model on the support set
            adapted_model = self.adapt(support_batch, inner_steps)

            # Compute loss on the query set with the adapted model
            query_loss = self._compute_loss(adapted_model, query_batch)
            return query_loss

        else:
            # Full MAML with second derivatives
            # Create a differentiable version of the model
            with higher.innerloop_ctx(
                self.model,
                torch.optim.SGD(self.model.get_adapter_parameters(self.adapter_name), lr=self.inner_lr),
                copy_initial_weights=False
            ) as (fmodel, diffopt):
                # Inner loop adaptation on support set
                for _ in range(inner_steps):
                    support_loss = self._compute_loss(fmodel, support_batch)
                    diffopt.step(support_loss)

                # Compute loss on query set with adapted model
                query_loss = self._compute_loss(fmodel, query_batch)
                return query_loss
        """


class MeLPA(nn.Module):
    """
    Meta-Learned Personalized Adapters framework.
    Combines the adapter-augmented foundation model with meta-learning components.
    """
    def __init__(
        self, 
        base_model, 
        adapter_config, 
        init_network_config=None,
        update_mechanism_config=None
    ):
        super().__init__()
        
        # Create model with adapters
        self.model = TransformerWithAdapters(base_model, adapter_config)
        
        # Initialize meta-learning components
        self.init_network = None
        self.update_mechanism = None
        
        # If configs provided, create the meta-learning components
        if init_network_config is not None:
            # Extract adapter shapes from a temporary adapter
            temp_adapter_name = "__temp_adapter__"
            self.model.add_adapter(temp_adapter_name)
            adapter_shapes = self._get_adapter_shapes(temp_adapter_name)
            
            # Create initialization network
            self.init_network = InitializationNetwork(
                input_dim=init_network_config.get("input_dim", 768),
                adapter_shapes=adapter_shapes,
                hidden_dims=init_network_config.get("hidden_dims", None),
                use_task_context=init_network_config.get("use_task_context", True)
            )
        
        if update_mechanism_config is not None:
            # Create update mechanism
            if update_mechanism_config.get("method", "learned_lr") == "learned_lr":
                # Get parameter names for the learned learning rates
                if not hasattr(self, "adapter_shapes"):
                    temp_adapter_name = "__temp_adapter__"
                    self.model.add_adapter(temp_adapter_name)
                    adapter_shapes = self._get_adapter_shapes(temp_adapter_name)
                
                param_names = list(adapter_shapes.keys())
                self.update_mechanism = UpdateMechanism(
                    method="learned_lr", 
                    param_names=param_names
                )
            else:
                self.update_mechanism = UpdateMechanism(
                    method=update_mechanism_config.get("method", "mlp_update"),
                    input_dim=update_mechanism_config.get("input_dim", 2),
                    hidden_dims=update_mechanism_config.get("hidden_dims", [64, 32])
                )
    
    def _get_adapter_shapes(self, adapter_name):
        """Extract shapes of adapter parameters."""
        adapter_shapes = {}
        for name, param in self.model.named_parameters():
            if adapter_name in name and param.requires_grad:
                adapter_shapes[name] = param.shape
        return adapter_shapes
    
    def initialize_adapter(self, adapter_name, task_context=None):
        """
        Initialize adapter parameters using the meta-learned initialization network.
        
        Args:
            adapter_name: Name of the adapter to initialize
            task_context: Optional tensor encoding task context information
        """
        if self.init_network is None:
            raise ValueError("Initialization network not set up. Provide init_network_config when creating MeLPA.")
        
        # Add a new adapter with default initialization
        self.model.add_adapter(adapter_name)
        
        # Generate initial parameters using the initialization network
        initial_params = self.init_network(task_context)
        
        # Apply these parameters to the adapter
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if adapter_name in name and name in initial_params:
                    param.copy_(initial_params[name])
    
    def update_adapter(self, adapter_name, batch):
        """
        Update adapter parameters using the meta-learned update mechanism.
        
        Args:
            adapter_name: Name of the adapter to update
            batch: Batch of data for computing gradients
        """
        if self.update_mechanism is None:
            raise ValueError("Update mechanism not set up. Provide update_mechanism_config when creating MeLPA.")
        
        # Get adapter parameters
        adapter_params = {}
        for name, param in self.model.named_parameters():
            if adapter_name in name:
                adapter_params[name] = param
        
        # Compute gradients
        self.model.zero_grad()
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        outputs = self.model(inputs, adapter_name=adapter_name)
        loss = F.cross_entropy(outputs.logits, batch['labels'])
        loss.backward()
        
        # Collect gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if adapter_name in name and param.grad is not None:
                gradients[name] = param.grad.clone()
        
        # Compute updates using the update mechanism
        updates = self.update_mechanism(adapter_params, gradients)
        
        # Apply updates
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in updates:
                    param.add_(updates[name])
    
    def meta_train_step(self, support_batch, query_batch, adapter_name):
        """
        Perform a meta-training step using a support and query batch.
        
        Args:
            support_batch: Batch of support set data
            query_batch: Batch of query set data
            adapter_name: Name of the adapter to use
        
        Returns:
            Loss on the query set after adaptation
        """
        # Initialize MAML with our model
        maml = MAML(
            model=self.model,
            adapter_name=adapter_name,
            inner_lr=0.01,  # Could be made a hyperparameter
            first_order=True  # Using first-order approximation for efficiency
        )
        
        # Perform MAML forward pass
        query_loss = maml(support_batch, query_batch)
        
        return query_loss
    
    def forward(self, inputs, adapter_name=None):
        """Forward pass to the model with the specified adapter."""
        return self.model(inputs, adapter_name)


# Import at the end to avoid circular imports
from models.adapters import TransformerWithAdapters