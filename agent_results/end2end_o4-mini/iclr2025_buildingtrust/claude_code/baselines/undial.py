"""
Implementation of the UNDIAL baseline method for LLM unlearning.
Based on: "UNDIAL: Self-Distillation with Adjusted Logits for Robust Unlearning in Large Language Models" (Dong et al., 2024)
"""

import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class LogitAdjuster(nn.Module):
    """
    Module for adjusting logits during self-distillation to selectively reduce
    the influence of targeted tokens.
    """
    
    def __init__(self, vocab_size, hidden_size=None, reduction_factor=0.5):
        """
        Initialize the logit adjuster.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Hidden size for projection (if None, no projection is used)
            reduction_factor: Factor for reducing token probabilities (0 to 1)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.reduction_factor = reduction_factor
        
        # Token-specific adjustment vectors
        self.token_adjustments = nn.Parameter(torch.zeros(vocab_size))
        
        # Optional projection layer
        self.use_projection = hidden_size is not None
        if self.use_projection:
            self.projection = nn.Linear(hidden_size, vocab_size)
            # Initialize to small values
            nn.init.normal_(self.projection.weight, std=0.01)
            nn.init.zeros_(self.projection.bias)
    
    def forward(self, logits, hidden_states=None, forget_tokens=None):
        """
        Adjust logits for unlearning.
        
        Args:
            logits: Original logits from the model
            hidden_states: Hidden states from the model (optional)
            forget_tokens: List of token indices to forget (if None, use learned adjustments)
            
        Returns:
            adjusted_logits: Adjusted logits
        """
        # Get basic shape
        batch_size = logits.size(0)
        seq_len = logits.size(1) if logits.dim() > 2 else 1
        
        # Reshape logits if needed
        if logits.dim() > 2:
            # [batch_size, seq_len, vocab_size]
            reshaped_logits = logits
        else:
            # [batch_size, vocab_size]
            reshaped_logits = logits.unsqueeze(1)
        
        # Apply token adjustments
        if forget_tokens is not None:
            # Create an adjustment tensor for specified tokens
            adjustment = torch.zeros_like(reshaped_logits)
            for token_idx in forget_tokens:
                adjustment[:, :, token_idx] = -self.reduction_factor * torch.abs(reshaped_logits[:, :, token_idx])
        else:
            # Use learned adjustments
            adjustment = self.token_adjustments.view(1, 1, -1).expand_as(reshaped_logits)
        
        # Apply optional projection from hidden states
        if self.use_projection and hidden_states is not None:
            # Project hidden states to vocab space
            if hidden_states.dim() == 2:
                # [batch_size, hidden_size]
                projected = self.projection(hidden_states).unsqueeze(1)
            else:
                # [batch_size, seq_len, hidden_size]
                projected = self.projection(hidden_states)
                
            # Add projection to adjustment
            adjustment = adjustment + projected
        
        # Apply adjustments
        adjusted_logits = reshaped_logits + adjustment
        
        # Reshape back if needed
        if logits.dim() <= 2:
            adjusted_logits = adjusted_logits.squeeze(1)
        
        return adjusted_logits


class UNDIALMethod:
    """
    UNDIAL unlearning method implementation.
    
    The method uses self-distillation with adjusted logits to selectively reduce
    the influence of targeted tokens.
    """
    
    def __init__(
        self,
        model,
        optimizer_class=torch.optim.AdamW,
        learning_rate=5e-5,
        num_epochs=3,
        batch_size=16,
        distillation_temp=2.0,
        reduction_factor=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the UNDIAL method.
        
        Args:
            model: The pretrained LLM model
            optimizer_class: Optimizer class to use
            learning_rate: Learning rate for fine-tuning
            num_epochs: Number of epochs for fine-tuning
            batch_size: Batch size for fine-tuning
            distillation_temp: Temperature for distillation
            reduction_factor: Factor for reducing token probabilities
            device: Device to use for computation
        """
        self.model = model.to(device)
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.distillation_temp = distillation_temp
        self.reduction_factor = reduction_factor
        self.device = device
        
        # Determine vocabulary size
        if hasattr(model, 'config'):
            self.vocab_size = model.config.vocab_size
        else:
            # Try to infer from the model parameters
            for name, param in model.named_parameters():
                if 'lm_head' in name and 'weight' in name:
                    self.vocab_size = param.shape[0]
                    break
            else:
                self.vocab_size = 50257  # GPT-2 default vocab size
    
    def _extract_forget_tokens(self, deletion_set, top_k=100):
        """
        Extract the most frequent tokens from the deletion set.
        
        Args:
            deletion_set: Set of examples to delete
            top_k: Number of top tokens to consider
            
        Returns:
            forget_tokens: List of token indices to forget
        """
        # Count token occurrences
        token_counts = {}
        
        for example in deletion_set:
            # Extract input_ids
            if isinstance(example, dict) and 'input_ids' in example:
                input_ids = example['input_ids']
            elif isinstance(example, tuple) and isinstance(example[0], dict) and 'input_ids' in example[0]:
                input_ids = example[0]['input_ids']
            else:
                continue
                
            # Count tokens
            if isinstance(input_ids, torch.Tensor):
                tokens = input_ids.cpu().numpy().tolist()
            else:
                tokens = input_ids
                
            for token in tokens:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1
        
        # Get top-k tokens
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        forget_tokens = [token for token, count in sorted_tokens[:top_k]]
        
        return forget_tokens
    
    def _create_adjusted_model(self, model, hidden_size=None):
        """
        Create a copy of the model with a logit adjuster.
        
        Args:
            model: The model to copy
            hidden_size: Hidden size for projection (if None, no projection is used)
            
        Returns:
            adjusted_model: The model with logit adjuster
        """
        # Create a copy of the model
        adjusted_model = copy.deepcopy(model)
        
        # Create logit adjuster
        logit_adjuster = LogitAdjuster(
            self.vocab_size,
            hidden_size=hidden_size,
            reduction_factor=self.reduction_factor
        ).to(self.device)
        
        # Modify the forward method of the model
        original_forward = adjusted_model.forward
        
        def new_forward(input_ids=None, attention_mask=None, labels=None, forget_tokens=None, **kwargs):
            # Call the original forward method
            outputs = original_forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
            
            # Extract hidden states if available
            hidden_states = None
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]  # Use the last layer
            
            # Adjust logits
            adjusted_logits = logit_adjuster(outputs.logits, hidden_states, forget_tokens)
            
            # Create a new output with adjusted logits
            outputs.logits = adjusted_logits
            
            # Recompute loss if labels are provided
            if labels is not None:
                # Compute cross-entropy loss (similar to model.compute_loss)
                if adjusted_logits.shape[:-1] != labels.shape:
                    # Reshape for language modeling
                    shifted_logits = adjusted_logits.contiguous().view(-1, adjusted_logits.size(-1))
                    shifted_labels = labels.contiguous().view(-1)
                else:
                    shifted_logits = adjusted_logits
                    shifted_labels = labels
                    
                loss_fct = torch.nn.CrossEntropyLoss()
                outputs.loss = loss_fct(shifted_logits, shifted_labels)
            
            return outputs
        
        # Replace the forward method
        adjusted_model.forward = new_forward
        
        # Add the logit adjuster to the model
        adjusted_model.logit_adjuster = logit_adjuster
        
        return adjusted_model
    
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
        
        # Combine losses (0.5 weight for each)
        loss = 0.5 * kl_loss + 0.5 * ce_loss
        
        return loss
    
    def _train_step(self, batch, teacher_model, student_model, forget_tokens=None, is_forget_batch=False):
        """
        Perform a single training step.
        
        Args:
            batch: Batch of examples
            teacher_model: Teacher model (original)
            student_model: Student model (with logit adjuster)
            forget_tokens: List of token indices to forget
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
        
        # Forward pass for student
        student_outputs = student_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            forget_tokens=forget_tokens if is_forget_batch else None
        )
        student_logits = student_outputs.logits
        
        # Compute loss
        loss = self._compute_distillation_loss(student_logits, teacher_logits, targets)
        
        return loss
    
    def unlearn(self, validation_data, deletion_set, hidden_size=None):
        """
        Perform unlearning using the UNDIAL method.
        
        Args:
            validation_data: Dataset of examples to retain
            deletion_set: Set of examples to delete
            hidden_size: Hidden size for projection (if None, no projection is used)
            
        Returns:
            unlearned_model: The unlearned model
            metrics: Unlearning metrics
        """
        # Start timing
        start_time = time.time()
        
        # Extract tokens to forget
        forget_tokens = self._extract_forget_tokens(deletion_set)
        
        # Create student model with logit adjuster
        unlearned_model = self._create_adjusted_model(self.model, hidden_size)
        
        # Create data loaders
        validation_loader = DataLoader(
            validation_data,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        deletion_loader = DataLoader(
            deletion_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Create optimizer (only for student model)
        optimizer = self.optimizer_class(
            [p for n, p in unlearned_model.named_parameters() if 'logit_adjuster' in n],
            lr=self.learning_rate
        )
        
        # Train the student model
        unlearned_model.train()
        self.model.eval()  # Teacher model (original) is always in eval mode
        
        for epoch in range(self.num_epochs):
            # Train on validation data (to retain information)
            for batch in validation_loader:
                optimizer.zero_grad()
                loss = self._train_step(
                    batch, self.model, unlearned_model,
                    forget_tokens=None, is_forget_batch=False
                )
                loss.backward()
                optimizer.step()
            
            # Train on deletion data (to forget information)
            for batch in deletion_loader:
                optimizer.zero_grad()
                loss = self._train_step(
                    batch, self.model, unlearned_model,
                    forget_tokens=forget_tokens, is_forget_batch=True
                )
                loss.backward()
                optimizer.step()
        
        # Fine-tune the full model with frozen logit adjuster
        if epoch >= self.num_epochs - 1:
            # Freeze logit adjuster
            for param in unlearned_model.logit_adjuster.parameters():
                param.requires_grad = False
                
            # Create a new optimizer for the rest of the model
            fine_tune_optimizer = self.optimizer_class(
                [p for n, p in unlearned_model.named_parameters() if 'logit_adjuster' not in n],
                lr=self.learning_rate * 0.1  # Lower learning rate for fine-tuning
            )
            
            # Fine-tune for one epoch
            for batch in validation_loader:
                fine_tune_optimizer.zero_grad()
                loss = self._train_step(
                    batch, self.model, unlearned_model,
                    forget_tokens=None, is_forget_batch=False
                )
                loss.backward()
                fine_tune_optimizer.step()
        
        # Compute metrics
        metrics = {
            'method': 'UNDIAL',
            'compute_time': time.time() - start_time,
            'num_forget_tokens': len(forget_tokens)
        }
        
        return unlearned_model, metrics