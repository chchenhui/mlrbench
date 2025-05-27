"""
Implementation of the ReLearn baseline method for LLM unlearning.
Based on: "ReLearn: Unlearning via Learning for Large Language Models" (Xu et al., 2025)
"""

import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class AugmentedDataset(Dataset):
    """
    Dataset for the ReLearn method, which augments the deletion set with synthetic data.
    """
    
    def __init__(self, original_data, deletion_set, tokenizer=None, augment_factor=5):
        """
        Initialize the dataset.
        
        Args:
            original_data: Dataset of examples to keep
            deletion_set: Set of examples to forget
            tokenizer: Tokenizer for text augmentation
            augment_factor: Number of augmentations per deletion example
        """
        self.original_data = original_data
        self.deletion_set = deletion_set
        self.tokenizer = tokenizer
        self.augment_factor = augment_factor
        
        # Create augmented deletion set
        self.augmented_deletion = self._augment_deletion_set()
    
    def _augment_deletion_set(self):
        """
        Augment the deletion set with synthetic data.
        
        Returns:
            augmented_deletion (list): Augmented deletion set
        """
        augmented_deletion = []
        
        for example in self.deletion_set:
            # Add the original example
            augmented_deletion.append(example)
            
            # Add augmented examples
            augmented_examples = self._create_augmentations(example)
            augmented_deletion.extend(augmented_examples)
        
        return augmented_deletion
    
    def _create_augmentations(self, example):
        """
        Create augmented versions of an example.
        
        Args:
            example: Example to augment
            
        Returns:
            augmented_examples (list): Augmented examples
        """
        augmented_examples = []
        
        # If we have a tokenizer, use it for augmentation
        if self.tokenizer is not None:
            # Extract text from the example
            if isinstance(example, dict) and 'input_ids' in example:
                # Decode the input_ids to get the text
                text = self.tokenizer.decode(example['input_ids'])
            elif isinstance(example, tuple) and isinstance(example[0], dict) and 'input_ids' in example[0]:
                # Decode the input_ids from the first element
                text = self.tokenizer.decode(example[0]['input_ids'])
            else:
                # Default to using the example as is
                text = str(example)
            
            # Create augmentations by:
            # 1. Word replacement
            for _ in range(self.augment_factor // 2):
                # Tokenize and replace random tokens
                tokens = text.split()
                for i in range(min(5, len(tokens))):
                    idx = torch.randint(len(tokens), (1,)).item()
                    tokens[idx] = "[MASK]"  # Replace with mask token
                
                # Create new example with augmented text
                augmented_text = " ".join(tokens)
                augmented_example = self._create_example_from_text(augmented_text, example)
                augmented_examples.append(augmented_example)
            
            # 2. Random deletion
            for _ in range(self.augment_factor - self.augment_factor // 2):
                # Tokenize and delete random tokens
                tokens = text.split()
                for i in range(min(3, len(tokens))):
                    idx = torch.randint(len(tokens), (1,)).item()
                    tokens.pop(idx)
                
                # Create new example with augmented text
                augmented_text = " ".join(tokens)
                augmented_example = self._create_example_from_text(augmented_text, example)
                augmented_examples.append(augmented_example)
        else:
            # Without a tokenizer, just duplicate the example
            augmented_examples = [copy.deepcopy(example) for _ in range(self.augment_factor)]
        
        return augmented_examples
    
    def _create_example_from_text(self, text, template_example):
        """
        Create a new example from augmented text using the original as a template.
        
        Args:
            text: Augmented text
            template_example: Original example to use as a template
            
        Returns:
            new_example: New example with augmented text
        """
        # Tokenize the text
        encoded = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create a new example with the same structure as the template
        if isinstance(template_example, dict):
            new_example = {
                'input_ids': encoded['input_ids'][0],
                'attention_mask': encoded['attention_mask'][0],
                'targets': encoded['input_ids'][0][1:]  # Shifted for language modeling
            }
        elif isinstance(template_example, tuple) and len(template_example) == 2:
            # If the template is a tuple of (inputs, targets)
            new_example = (
                {
                    'input_ids': encoded['input_ids'][0],
                    'attention_mask': encoded['attention_mask'][0]
                },
                encoded['input_ids'][0][1:]  # Shifted for language modeling
            )
        else:
            # Default fallback
            new_example = encoded
        
        return new_example
    
    def __len__(self):
        return len(self.original_data) + len(self.augmented_deletion)
    
    def __getitem__(self, idx):
        if idx < len(self.original_data):
            return self.original_data[idx]
        else:
            return self.augmented_deletion[idx - len(self.original_data)]


class RelearnUnlearningMethod:
    """
    ReLearn unlearning method implementation.
    
    The method works by creating an augmented dataset with modified versions of the 
    deletion set, and then fine-tuning the model to "forget" these examples.
    """
    
    def __init__(
        self,
        model,
        optimizer_class=torch.optim.AdamW,
        learning_rate=5e-5,
        num_epochs=3,
        batch_size=16,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the ReLearn method.
        
        Args:
            model: The pretrained LLM model
            optimizer_class: Optimizer class to use
            learning_rate: Learning rate for fine-tuning
            num_epochs: Number of epochs for fine-tuning
            batch_size: Batch size for fine-tuning
            device: Device to use for computation
        """
        self.model = model.to(device)
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
    
    def _compute_unlikelihood_loss(self, logits, targets, attention_mask=None):
        """
        Compute unlikelihood loss for encouraging the model to avoid generating
        specific outputs.
        
        Args:
            logits: Model logits
            targets: Target tokens
            attention_mask: Attention mask
            
        Returns:
            loss: Unlikelihood loss
        """
        # Calculate standard cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='none'
        ).view(targets.shape)
        
        # Apply mask if provided
        if attention_mask is not None:
            ce_loss = ce_loss * attention_mask
        
        # Calculate unlikelihood loss components
        probs = F.softmax(logits, dim=-1)
        one_hot = F.one_hot(targets, num_classes=logits.size(-1))
        
        # Unlikelihood term: maximize probability of not generating the target
        neg_probs = probs * (1 - one_hot)
        ul_loss = -torch.log(1 - neg_probs.sum(dim=-1) + 1e-10)
        
        # Apply mask if provided
        if attention_mask is not None:
            ul_loss = ul_loss * attention_mask
        
        # Combine losses
        combined_loss = ce_loss + ul_loss
        
        # Average over non-padding positions
        if attention_mask is not None:
            combined_loss = combined_loss.sum() / attention_mask.sum()
        else:
            combined_loss = combined_loss.mean()
        
        return combined_loss
    
    def _train_step(self, batch, is_forget_batch=False):
        """
        Perform a single training step.
        
        Args:
            batch: Batch of examples
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
        
        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Compute loss based on whether this is a forget batch
        if is_forget_batch:
            # For examples to forget, use unlikelihood loss
            loss = self._compute_unlikelihood_loss(
                logits, targets, inputs.get('attention_mask')
            )
        else:
            # For examples to retain, use standard cross-entropy loss
            if logits.shape[:-1] != targets.shape:
                # Reshape for language modeling
                shifted_logits = logits.contiguous().view(-1, logits.size(-1))
                shifted_targets = targets.contiguous().view(-1)
            else:
                shifted_logits = logits
                shifted_targets = targets
                
            loss = F.cross_entropy(shifted_logits, shifted_targets)
        
        return loss
    
    def unlearn(self, validation_data, deletion_set, tokenizer=None):
        """
        Perform unlearning using the ReLearn method.
        
        Args:
            validation_data: Dataset of examples to retain
            deletion_set: Set of examples to forget
            tokenizer: Tokenizer for text augmentation
            
        Returns:
            unlearned_model: The unlearned model
            metrics: Unlearning metrics
        """
        # Start timing
        start_time = time.time()
        
        # Create a copy of the model for unlearning
        unlearned_model = copy.deepcopy(self.model)
        
        # Create augmented dataset
        augmented_dataset = AugmentedDataset(
            validation_data, deletion_set, tokenizer=tokenizer
        )
        
        # Create data loaders
        forget_indices = list(range(len(validation_data), len(augmented_dataset)))
        retain_indices = list(range(len(validation_data)))
        
        from torch.utils.data import Subset
        forget_loader = DataLoader(
            Subset(augmented_dataset, forget_indices),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        retain_loader = DataLoader(
            Subset(augmented_dataset, retain_indices),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Create optimizer
        optimizer = self.optimizer_class(
            unlearned_model.parameters(),
            lr=self.learning_rate
        )
        
        # Train the model
        unlearned_model.train()
        
        for epoch in range(self.num_epochs):
            # Train on examples to retain
            for batch in retain_loader:
                optimizer.zero_grad()
                loss = self._train_step(batch, is_forget_batch=False)
                loss.backward()
                optimizer.step()
            
            # Train on examples to forget with unlikelihood loss
            for batch in forget_loader:
                optimizer.zero_grad()
                loss = self._train_step(batch, is_forget_batch=True)
                loss.backward()
                optimizer.step()
        
        # Compute metrics
        metrics = {
            'method': 'ReLearn',
            'compute_time': time.time() - start_time
        }
        
        return unlearned_model, metrics