"""
Implementation of the "Unlearn What You Want to Forget" baseline method for LLM unlearning.
Based on: "Unlearn What You Want to Forget: Efficient Unlearning for LLMs" (Chen & Yang, 2023)
"""

import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class UnlearningLayer(nn.Module):
    """
    Lightweight unlearning layer that can be inserted into a transformer architecture.
    """
    
    def __init__(self, hidden_size, dropout_prob=0.1):
        """
        Initialize the unlearning layer.
        
        Args:
            hidden_size: Hidden size of the transformer
            dropout_prob: Dropout probability
        """
        super().__init__()
        
        # Layer components
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Add a gate mechanism to control information flow
        self.gate = nn.Linear(hidden_size, 1)
        
        # Initialize parameters
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.ones_(self.gate.bias)  # Initialize to pass through most information
    
    def forward(self, hidden_states):
        """
        Forward pass.
        
        Args:
            hidden_states: Hidden states from the transformer layer
            
        Returns:
            modified_hidden_states: Modified hidden states
        """
        # Calculate gate values (sigmoid to get values between 0 and 1)
        gate_values = torch.sigmoid(self.gate(hidden_states))
        
        # Apply transformation
        transformed = self.dense(hidden_states)
        transformed = F.gelu(transformed)
        transformed = self.dropout(transformed)
        
        # Apply layer normalization
        transformed = self.layer_norm(transformed)
        
        # Apply gate to control information flow
        # gate_values close to 0 will block information, close to 1 will pass it
        modified_hidden_states = gate_values * transformed + (1 - gate_values) * hidden_states
        
        return modified_hidden_states


class UnlearnWhatYouWantMethod:
    """
    Unlearn What You Want method implementation.
    
    The method works by inserting lightweight unlearning layers into the transformer
    architecture and then fine-tuning these layers with a teacher-student objective.
    """
    
    def __init__(
        self,
        model,
        optimizer_class=torch.optim.AdamW,
        learning_rate=1e-4,
        num_epochs=3,
        batch_size=16,
        distillation_temp=2.0,
        distillation_alpha=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the method.
        
        Args:
            model: The pretrained LLM model
            optimizer_class: Optimizer class to use
            learning_rate: Learning rate for fine-tuning
            num_epochs: Number of epochs for fine-tuning
            batch_size: Batch size for fine-tuning
            distillation_temp: Temperature for distillation
            distillation_alpha: Weight for distillation loss
            device: Device to use for computation
        """
        self.model = model.to(device)
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.distillation_temp = distillation_temp
        self.distillation_alpha = distillation_alpha
        self.device = device
        self.unlearning_layers = []
    
    def _insert_unlearning_layers(self, model):
        """
        Insert unlearning layers into the model.
        
        Args:
            model: The model to modify
            
        Returns:
            modified_model: The modified model with unlearning layers
        """
        # Create a copy of the model
        modified_model = copy.deepcopy(model)
        
        # Determine the hidden size based on the model architecture
        if hasattr(modified_model, 'config'):
            hidden_size = modified_model.config.hidden_size
        else:
            # Try to infer from the model parameters
            for name, param in modified_model.named_parameters():
                if 'weight' in name and len(param.shape) == 2:
                    hidden_size = param.shape[0]
                    break
            else:
                hidden_size = 768  # Default size if we can't determine
        
        # Create unlearning layers
        self.unlearning_layers = []
        
        # For GPT-2 style models
        if hasattr(modified_model, 'transformer') and hasattr(modified_model.transformer, 'h'):
            num_layers = len(modified_model.transformer.h)
            
            # Insert unlearning layers after every 2 transformer layers
            for i in range(1, num_layers, 2):
                unlearning_layer = UnlearningLayer(hidden_size)
                unlearning_layer.to(self.device)
                self.unlearning_layers.append(unlearning_layer)
                
                # Save the original forward method
                original_forward = modified_model.transformer.h[i].forward
                
                # Define a new forward method that adds the unlearning layer
                def make_forward(idx, orig_forward):
                    def new_forward(hidden_states, *args, **kwargs):
                        # Call the original forward method
                        outputs = orig_forward(hidden_states, *args, **kwargs)
                        
                        # Apply unlearning layer to the output
                        if isinstance(outputs, tuple):
                            modified_outputs = (self.unlearning_layers[idx](outputs[0]),) + outputs[1:]
                        else:
                            modified_outputs = self.unlearning_layers[idx](outputs)
                        
                        return modified_outputs
                    
                    return new_forward
                
                # Replace the forward method
                layer_idx = (i - 1) // 2  # Index in unlearning_layers
                modified_model.transformer.h[i].forward = make_forward(layer_idx, original_forward)
        
        # For BERT style models
        elif hasattr(modified_model, 'encoder') and hasattr(modified_model.encoder, 'layer'):
            num_layers = len(modified_model.encoder.layer)
            
            # Insert unlearning layers after every 2 transformer layers
            for i in range(1, num_layers, 2):
                unlearning_layer = UnlearningLayer(hidden_size)
                unlearning_layer.to(self.device)
                self.unlearning_layers.append(unlearning_layer)
                
                # Save the original forward method
                original_forward = modified_model.encoder.layer[i].forward
                
                # Define a new forward method that adds the unlearning layer
                def make_forward(idx, orig_forward):
                    def new_forward(hidden_states, *args, **kwargs):
                        # Call the original forward method
                        outputs = orig_forward(hidden_states, *args, **kwargs)
                        
                        # Apply unlearning layer to the output
                        if isinstance(outputs, tuple):
                            modified_outputs = (self.unlearning_layers[idx](outputs[0]),) + outputs[1:]
                        else:
                            modified_outputs = self.unlearning_layers[idx](outputs)
                        
                        return modified_outputs
                    
                    return new_forward
                
                # Replace the forward method
                layer_idx = (i - 1) // 2  # Index in unlearning_layers
                modified_model.encoder.layer[i].forward = make_forward(layer_idx, original_forward)
        
        return modified_model
    
    def _compute_distillation_loss(self, student_logits, teacher_logits, targets):
        """
        Compute the distillation loss.
        
        Args:
            student_logits: Logits from the student model
            teacher_logits: Logits from the teacher model
            targets: Target tokens
            
        Returns:
            loss: Distillation loss
        """
        # KL divergence loss for distillation
        temp_scaled_student = student_logits / self.distillation_temp
        temp_scaled_teacher = teacher_logits / self.distillation_temp
        
        # Apply log_softmax and then compute KL divergence
        log_softmax_student = F.log_softmax(temp_scaled_student, dim=-1)
        softmax_teacher = F.softmax(temp_scaled_teacher, dim=-1)
        
        kl_loss = F.kl_div(
            log_softmax_student.view(-1, log_softmax_student.size(-1)),
            softmax_teacher.view(-1, softmax_teacher.size(-1)),
            reduction='batchmean'
        ) * (self.distillation_temp ** 2)
        
        # Cross-entropy loss with targets
        if student_logits.shape[:-1] != targets.shape:
            # Reshape for language modeling
            shifted_logits = student_logits.contiguous().view(-1, student_logits.size(-1))
            shifted_targets = targets.contiguous().view(-1)
        else:
            shifted_logits = student_logits
            shifted_targets = targets
            
        ce_loss = F.cross_entropy(shifted_logits, shifted_targets)
        
        # Combine losses
        loss = self.distillation_alpha * kl_loss + (1 - self.distillation_alpha) * ce_loss
        
        return loss
    
    def _compute_selective_distillation_loss(self, student_logits, teacher_logits, targets, is_forget_batch):
        """
        Compute selective distillation loss based on whether examples should be forgotten.
        
        Args:
            student_logits: Logits from the student model
            teacher_logits: Logits from the teacher model
            targets: Target tokens
            is_forget_batch: Whether this batch contains examples to forget
            
        Returns:
            loss: Selective distillation loss
        """
        if is_forget_batch:
            # For examples to forget, push student away from teacher
            # Invert the softmax distribution of the teacher
            temp_scaled_student = student_logits / self.distillation_temp
            temp_scaled_teacher = teacher_logits / self.distillation_temp
            
            softmax_teacher = F.softmax(temp_scaled_teacher, dim=-1)
            # Invert the teacher distribution (subtract from 1 and renormalize)
            inverted_teacher = 1.0 - softmax_teacher
            row_sums = inverted_teacher.sum(dim=-1, keepdim=True)
            inverted_teacher = inverted_teacher / row_sums
            
            # Ensure we don't have NaN values
            inverted_teacher = torch.nan_to_num(inverted_teacher, nan=1.0/inverted_teacher.size(-1))
            
            # Compute KL divergence with the inverted teacher
            log_softmax_student = F.log_softmax(temp_scaled_student, dim=-1)
            
            kl_loss = F.kl_div(
                log_softmax_student.view(-1, log_softmax_student.size(-1)),
                inverted_teacher.view(-1, inverted_teacher.size(-1)),
                reduction='batchmean'
            ) * (self.distillation_temp ** 2)
            
            # Also add unlikelihood loss for targets
            probs = F.softmax(student_logits, dim=-1)
            one_hot = F.one_hot(targets, num_classes=student_logits.size(-1))
            target_probs = torch.sum(probs * one_hot, dim=-1)
            unlikelihood_loss = -torch.log(1 - target_probs + 1e-10).mean()
            
            loss = kl_loss + unlikelihood_loss
        else:
            # For examples to retain, use standard distillation
            loss = self._compute_distillation_loss(student_logits, teacher_logits, targets)
        
        return loss
    
    def _train_step(self, batch, teacher_model, student_model, is_forget_batch=False):
        """
        Perform a single training step.
        
        Args:
            batch: Batch of examples
            teacher_model: Teacher model (original)
            student_model: Student model (with unlearning layers)
            is_forget_batch: Whether this batch contains examples to forget
            
        Returns:
            loss: Training loss
        """
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
        
        # Get teacher logits
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Get student logits
        student_outputs = student_model(**inputs)
        student_logits = student_outputs.logits
        
        # Compute loss
        loss = self._compute_selective_distillation_loss(
            student_logits, teacher_logits, targets, is_forget_batch
        )
        
        return loss
    
    def unlearn(self, validation_data, deletion_set):
        """
        Perform unlearning using the Unlearn What You Want method.
        
        Args:
            validation_data: Dataset of examples to retain
            deletion_set: Set of examples to forget
            
        Returns:
            unlearned_model: The unlearned model
            metrics: Unlearning metrics
        """
        # Start timing
        start_time = time.time()
        
        # Insert unlearning layers into the model
        unlearned_model = self._insert_unlearning_layers(self.model)
        
        # Freeze the original model parameters, only train the unlearning layers
        for name, param in unlearned_model.named_parameters():
            if not any(f"unlearning_layers.{i}" in name for i in range(len(self.unlearning_layers))):
                param.requires_grad = False
        
        # Create data loaders
        retain_loader = DataLoader(
            validation_data,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        forget_loader = DataLoader(
            deletion_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Create optimizer (only for unlearning layers)
        optimizer = self.optimizer_class(
            [param for param in unlearned_model.parameters() if param.requires_grad],
            lr=self.learning_rate
        )
        
        # Train the unlearning layers
        unlearned_model.train()
        self.model.eval()  # Teacher model (original) is always in eval mode
        
        for epoch in range(self.num_epochs):
            # Train on examples to retain
            for batch in retain_loader:
                optimizer.zero_grad()
                loss = self._train_step(batch, self.model, unlearned_model, is_forget_batch=False)
                loss.backward()
                optimizer.step()
            
            # Train on examples to forget
            for batch in forget_loader:
                optimizer.zero_grad()
                loss = self._train_step(batch, self.model, unlearned_model, is_forget_batch=True)
                loss.backward()
                optimizer.step()
        
        # Compute metrics
        metrics = {
            'method': 'Unlearn What You Want',
            'compute_time': time.time() - start_time,
            'num_unlearning_layers': len(self.unlearning_layers)
        }
        
        return unlearned_model, metrics