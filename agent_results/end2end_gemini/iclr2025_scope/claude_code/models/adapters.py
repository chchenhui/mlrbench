"""
Implementation of adapter modules for foundation models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation


class PfeifferAdapter(nn.Module):
    """
    Pfeiffer-style adapter module with bottleneck architecture.
    Adapters are inserted after the attention and feed-forward blocks in transformer layers.
    """
    def __init__(self, input_dim, bottleneck_dim, activation="gelu", init_scale=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Down projection
        self.down = nn.Linear(input_dim, bottleneck_dim)
        # Up projection
        self.up = nn.Linear(bottleneck_dim, input_dim)
        
        # Activation function
        self.activation = get_activation(activation)
        
        # Initialize with small weights
        nn.init.normal_(self.down.weight, std=init_scale)
        nn.init.normal_(self.up.weight, std=init_scale)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)
    
    def forward(self, x):
        residual = x
        x = self.down(x)
        x = self.activation(x)
        x = self.up(x)
        return x + residual


class LoRAAdapter(nn.Module):
    """
    LoRA-style adapter that uses low-rank matrices to adapt the model.
    """
    def __init__(self, input_dim, output_dim, rank, alpha=1.0, init_scale=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.A = nn.Linear(input_dim, rank, bias=False)
        self.B = nn.Linear(rank, output_dim, bias=False)
        
        # Initialize with small weights
        nn.init.normal_(self.A.weight, std=init_scale)
        nn.init.zeros_(self.B.weight)
    
    def forward(self, x):
        # Original input preserved as residual
        return x + self.B(self.A(x)) * self.scaling


class AdapterController(nn.Module):
    """
    Controls multiple adapters for different tasks/users.
    """
    def __init__(self, adapter_config):
        super().__init__()
        self.adapters = nn.ModuleDict({})
        self.adapter_config = adapter_config
    
    def add_adapter(self, adapter_name, adapter_type="pfeiffer", **kwargs):
        """Add a new adapter for a task/user."""
        # Ensure adapter_name is safe for ModuleDict keys (no dots)
        safe_name = adapter_name.replace(".", "_")

        # Check if adapter already exists
        if safe_name in self.adapters:
            return

        # Create the adapter based on type
        if adapter_type.lower() == "pfeiffer":
            adapter = PfeifferAdapter(
                input_dim=kwargs.get("input_dim", self.adapter_config["input_dim"]),
                bottleneck_dim=kwargs.get("bottleneck_dim", self.adapter_config["bottleneck_dim"]),
                activation=kwargs.get("activation", self.adapter_config.get("activation", "gelu")),
                init_scale=kwargs.get("init_scale", self.adapter_config.get("init_scale", 1e-3))
            )
        elif adapter_type.lower() == "lora":
            adapter = LoRAAdapter(
                input_dim=kwargs.get("input_dim", self.adapter_config["input_dim"]),
                output_dim=kwargs.get("output_dim", self.adapter_config["output_dim"]),
                rank=kwargs.get("rank", self.adapter_config["rank"]),
                alpha=kwargs.get("alpha", self.adapter_config.get("alpha", 1.0)),
                init_scale=kwargs.get("init_scale", self.adapter_config.get("init_scale", 1e-3))
            )
        else:
            raise ValueError(f"Adapter type {adapter_type} not supported")

        # Register the adapter with safe name
        self.adapters[safe_name] = adapter
    
    def get_adapter(self, adapter_name):
        """Get a specific adapter by name."""
        # Convert to safe name for lookup
        safe_name = adapter_name.replace(".", "_")
        if safe_name not in self.adapters:
            raise ValueError(f"Adapter {adapter_name} not found")
        return self.adapters[safe_name]

    def forward(self, x, adapter_name):
        """Forward pass using the specified adapter."""
        # Convert to safe name for lookup
        safe_name = adapter_name.replace(".", "_")
        if safe_name not in self.adapters:
            raise ValueError(f"Adapter {adapter_name} not found")
        return self.adapters[safe_name](x)


class TransformerWithAdapters(nn.Module):
    """
    Transformer model with adapters integrated.
    This is a simplified implementation for demonstration purposes.
    Real implementation would involve modifying attention and FFN blocks in transformer layers.
    """
    def __init__(self, base_model, adapter_config):
        super().__init__()
        self.base_model = base_model
        self.adapter_config = adapter_config
        self.adapter_controllers = self._create_adapter_controllers()
        self._freeze_base_model()
    
    def _freeze_base_model(self):
        """Freeze the base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def _create_adapter_controllers(self):
        """
        Create adapter controllers for each layer in the transformer.
        This is a simplified implementation.
        """
        # This would need to be customized based on the specific base model architecture
        adapter_controllers = nn.ModuleDict({})

        # Example for a model like BERT or GPT with layers and attention/ffn blocks
        if hasattr(self.base_model, "encoder") and hasattr(self.base_model.encoder, "layer"):
            # BERT-like models
            for i, _ in enumerate(self.base_model.encoder.layer):
                # Add attention output adapter
                adapter_controllers[f"layer_{i}_attention"] = AdapterController(self.adapter_config)
                # Add FFN output adapter
                adapter_controllers[f"layer_{i}_ffn"] = AdapterController(self.adapter_config)
        elif hasattr(self.base_model, "h"):
            # GPT-like models
            for i, _ in enumerate(self.base_model.h):
                # Add attention output adapter
                adapter_controllers[f"layer_{i}_attention"] = AdapterController(self.adapter_config)
                # Add FFN output adapter
                adapter_controllers[f"layer_{i}_ffn"] = AdapterController(self.adapter_config)
        elif hasattr(self.base_model, "distilbert") and hasattr(self.base_model.distilbert, "transformer") and hasattr(self.base_model.distilbert.transformer, "layer"):
            # DistilBERT models
            for i, _ in enumerate(self.base_model.distilbert.transformer.layer):
                # Add attention output adapter
                adapter_controllers[f"layer_{i}_attention"] = AdapterController(self.adapter_config)
                # Add FFN output adapter
                adapter_controllers[f"layer_{i}_ffn"] = AdapterController(self.adapter_config)
        else:
            # Fallback for unsupported architectures - create at least one adapter
            # This is a simplified approach for demonstration/experiments
            adapter_controllers["fallback_adapter"] = AdapterController(self.adapter_config)
            print("Warning: Using fallback adapter for unsupported model architecture")

        return adapter_controllers
    
    def add_adapter(self, adapter_name, adapter_type="pfeiffer"):
        """Add a new adapter for all layers."""
        for controller in self.adapter_controllers.values():
            controller.add_adapter(adapter_name, adapter_type)
    
    def get_adapter_parameters(self, adapter_name):
        """Get parameters of a specific adapter across all layers."""
        params = []
        # Convert to safe name for lookup
        safe_adapter_name = adapter_name.replace(".", "_")
        for controller in self.adapter_controllers.values():
            if safe_adapter_name in controller.adapters:
                params.extend(controller.adapters[safe_adapter_name].parameters())
        return params
    
    def forward(self, inputs=None, adapter_name=None, **kwargs):
        """
        Forward pass with adapter integration.
        This is a simplified implementation and would need to be customized for the specific model.

        Args:
            inputs: Dictionary of input tensors or keyword arguments
            adapter_name: Name of the adapter to use
            **kwargs: Additional arguments to pass to the base model
        """
        # Combine inputs dict with kwargs if both provided
        if inputs is not None:
            combined_inputs = inputs.copy() if isinstance(inputs, dict) else {"inputs": inputs}
            combined_inputs.update(kwargs)
        else:
            combined_inputs = kwargs

        # If no adapter specified or adapters not enabled, use the base model directly
        if adapter_name is None:
            return self.base_model(**combined_inputs)

        # Check if adapter exists in all controllers
        safe_adapter_name = adapter_name.replace(".", "_")
        for controller in self.adapter_controllers.values():
            if safe_adapter_name not in controller.adapters:
                raise ValueError(f"Adapter {adapter_name} not found in all controllers")

        # This is a placeholder. In a real implementation, you would need to:
        # 1. Modify the base model's forward pass to include adapter interventions
        # 2. Insert adapters after attention and FFN blocks in each layer
        # The actual implementation depends heavily on the base model's architecture

        # Placeholder: Forward pass with adapters
        outputs = self.base_model(**combined_inputs)

        return outputs