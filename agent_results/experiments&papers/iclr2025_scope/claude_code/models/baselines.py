#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline models for comparison with the proposed architecture.

This module implements various baseline models for comparison:
1. Standard Transformer with varying context windows
2. Traditional RAG approaches with naive concatenation
3. Recent efficient attention methods (AttentionRAG, GCA)
4. KV cache compression techniques (RazorAttention, PyramidKV)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import namedtuple


class StandardTransformerAttention(nn.Module):
    """
    Standard multi-head attention implementation for the transformer.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute standard multi-head attention.
        
        Args:
            query: Tensor of shape [batch_size, query_length, hidden_dim]
            key: Tensor of shape [batch_size, key_length, hidden_dim]
            value: Tensor of shape [batch_size, key_length, hidden_dim]
            attention_mask: Optional tensor of shape [batch_size, query_length, key_length]
                          for masking attention weights
        
        Returns:
            output: Tensor of shape [batch_size, query_length, hidden_dim]
            attention_weights: Tensor of shape [batch_size, num_heads, query_length, key_length]
        """
        batch_size, query_length, _ = query.shape
        _, key_length, _ = key.shape
        
        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, query_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, key_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, key_length, self.num_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_length, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        # [batch_size, num_heads, query_length, key_length]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand the mask for all heads
            expanded_mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(~expanded_mask.bool(), float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        # [batch_size, num_heads, query_length, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose and reshape
        # [batch_size, query_length, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # [batch_size, query_length, hidden_dim]
        attn_output = attn_output.view(batch_size, query_length, self.hidden_dim)
        
        # Apply output projection
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class StandardTransformerLayer(nn.Module):
    """
    Standard transformer encoder layer.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = StandardTransformerAttention(hidden_dim, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the transformer layer.
        
        Args:
            x: Tensor of shape [batch_size, seq_length, hidden_dim]
            attention_mask: Optional tensor for masking attention
        
        Returns:
            output: Tensor of shape [batch_size, seq_length, hidden_dim]
            attention_info: Dictionary containing attention weights
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        attn_output, attn_weights = self.attention(
            query=x,
            key=x,
            value=x,
            attention_mask=attention_mask
        )
        
        x = residual + self.dropout(attn_output)
        
        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        
        return x, {'attention_weights': attn_weights}


class StandardTransformer(nn.Module):
    """
    Standard Transformer model with full attention.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_sequence_length: int = 4096,
        pad_token_id: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        
        # Input projection
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            StandardTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Create an output type for structured returns
        self.ModelOutput = namedtuple(
            "ModelOutput",
            ["logits", "loss", "attention_weights"]
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Tuple, namedtuple]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_length] containing token ids
            attention_mask: Optional tensor of shape [batch_size, seq_length] indicating
                           valid tokens (1) and padding (0)
            labels: Optional tensor of shape [batch_size, seq_length] containing target ids
            return_dict: Whether to return a namedtuple
        
        Returns:
            If return_dict=True, a namedtuple containing:
                - logits: Tensor of shape [batch_size, seq_length, vocab_size]
                - loss: Loss value if labels are provided, otherwise None
                - attention_weights: List of attention weight tensors
            
            If return_dict=False, a tuple of:
                (logits, loss, attention_weights)
        """
        batch_size, seq_length = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Add positional embeddings
        positions = torch.arange(seq_length, device=input_ids.device).expand(batch_size, seq_length)
        pos_embeds = self.pos_embedding(positions)
        
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        
        # Project embeddings to hidden dimension
        hidden_states = self.input_projection(embeddings)
        
        # Prepare attention mask for self-attention
        attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
        attn_mask = attn_mask.expand(-1, -1, seq_length, -1)  # [batch_size, 1, seq_length, seq_length]
        
        # Process through transformer layers
        all_attention_weights = []
        
        for layer in self.layers:
            hidden_states, layer_outputs = layer(hidden_states, attention_mask=attn_mask)
            all_attention_weights.append(layer_outputs['attention_weights'])
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        if return_dict:
            return self.ModelOutput(
                logits=logits,
                loss=loss,
                attention_weights=all_attention_weights
            )
        else:
            return logits, loss, all_attention_weights
    
    def get_token_efficiency(self) -> float:
        """
        Get the token efficiency ratio.
        
        Returns:
            efficiency: Token efficiency ratio (always 1.0 for standard transformer)
        """
        return 1.0


class TraditionalRAGRetriever(nn.Module):
    """
    Simple retriever for traditional RAG that finds relevant documents via similarity.
    """
    def __init__(
        self,
        embedding_dim: int,
        retrieval_count: int = 3
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.retrieval_count = retrieval_count
        
        # Simple projection for query
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Cache for document embeddings
        self.register_buffer("doc_embeddings", torch.zeros(0, embedding_dim))
    
    def encode_documents(self, documents: List[str], tokenizer, model) -> None:
        """
        Encode and store document embeddings (simplified).
        
        Args:
            documents: List of document strings
            tokenizer: Tokenizer for encoding documents
            model: Model for generating embeddings
        """
        # This is a placeholder - in a real implementation, you would:
        # 1. Tokenize the documents
        # 2. Pass through the model to get embeddings
        # 3. Store the embeddings in the buffer
        
        # Here we just initialize random embeddings for simulation
        num_docs = len(documents)
        self.doc_embeddings = torch.randn(num_docs, self.embedding_dim)
        self.doc_embeddings = F.normalize(self.doc_embeddings, p=2, dim=-1)
    
    def forward(
        self,
        query_embeddings: torch.Tensor
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Retrieve relevant documents for the query.
        
        Args:
            query_embeddings: Tensor of shape [batch_size, seq_length, embedding_dim]
        
        Returns:
            retrieved_indices: List of indices of retrieved documents
            similarity_scores: Tensor of shape [batch_size, num_docs] with similarity scores
        """
        batch_size = query_embeddings.shape[0]
        
        # Average pool query embeddings
        query_pooled = torch.mean(query_embeddings, dim=1)  # [batch_size, embedding_dim]
        
        # Project query
        query_projected = self.query_projection(query_pooled)
        query_projected = F.normalize(query_projected, p=2, dim=-1)
        
        # Compute similarity scores
        # [batch_size, num_docs]
        similarity_scores = torch.matmul(query_projected, self.doc_embeddings.t())
        
        # Get top-k document indices
        _, retrieved_indices = torch.topk(
            similarity_scores, k=min(self.retrieval_count, self.doc_embeddings.size(0)), dim=-1
        )
        
        return retrieved_indices, similarity_scores


class TraditionalRAG(nn.Module):
    """
    Traditional Retrieval-Augmented Generation with naive document concatenation.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_sequence_length: int = 4096,
        retrieval_count: int = 3,
        pad_token_id: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length
        
        # Retriever component
        self.retriever = TraditionalRAGRetriever(embedding_dim, retrieval_count)
        
        # Base language model (standard transformer)
        self.language_model = StandardTransformer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            pad_token_id=pad_token_id,
            dropout=dropout
        )
        
        # Document store (simulated)
        self.documents = []
        self.doc_token_ids = []
        
        # Create an output type for structured returns
        self.ModelOutput = namedtuple(
            "ModelOutput",
            ["logits", "loss", "attention_weights", "retrieved_docs"]
        )
    
    def set_documents(
        self,
        documents: List[str],
        tokenizer
    ) -> None:
        """
        Set the document store and encode documents.
        
        Args:
            documents: List of document strings
            tokenizer: Tokenizer for encoding documents
        """
        self.documents = documents
        
        # Tokenize documents (simplified)
        self.doc_token_ids = []
        for doc in documents:
            # In a real implementation, this would be:
            # tokens = tokenizer.encode(doc, return_tensors="pt")
            # Here we just create random token IDs for simulation
            tokens = torch.randint(
                100, self.vocab_size, (1, min(100, self.max_sequence_length // 2))
            )
            self.doc_token_ids.append(tokens)
        
        # Encode documents for retrieval
        self.retriever.encode_documents(documents, tokenizer, self.language_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Tuple, namedtuple]:
        """
        Forward pass through the RAG model.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_length] containing token ids
            attention_mask: Optional tensor of shape [batch_size, seq_length] indicating
                           valid tokens (1) and padding (0)
            labels: Optional tensor of shape [batch_size, seq_length] containing target ids
            return_dict: Whether to return a namedtuple
        
        Returns:
            If return_dict=True, a namedtuple containing:
                - logits: Tensor of shape [batch_size, seq_length, vocab_size]
                - loss: Loss value if labels are provided, otherwise None
                - attention_weights: List of attention weight tensors
                - retrieved_docs: Indices of retrieved documents
            
            If return_dict=False, a tuple of:
                (logits, loss, attention_weights, retrieved_docs)
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Get token embeddings from the language model
        token_embeds = self.language_model.token_embedding(input_ids)
        
        # Retrieve relevant documents
        retrieved_indices, _ = self.retriever(token_embeds)
        
        # Concatenate input with retrieved documents
        augmented_ids = []
        augmented_masks = []
        
        for b in range(batch_size):
            # Get retrieved documents for this batch item
            batch_retrieved = []
            for idx in retrieved_indices[b]:
                idx = idx.item()
                if idx < len(self.doc_token_ids):
                    batch_retrieved.append(self.doc_token_ids[idx])
            
            # Concatenate document token IDs with input IDs
            batch_ids = input_ids[b]
            batch_mask = attention_mask[b]
            
            for doc_ids in batch_retrieved:
                # Move to the same device
                doc_ids = doc_ids.to(device)
                
                # Append document tokens
                batch_ids = torch.cat([doc_ids.squeeze(0), batch_ids])
                batch_mask = torch.cat([
                    torch.ones(doc_ids.size(1), device=device),
                    batch_mask
                ])
            
            # Truncate if too long
            if batch_ids.size(0) > self.max_sequence_length:
                batch_ids = batch_ids[-self.max_sequence_length:]
                batch_mask = batch_mask[-self.max_sequence_length:]
            
            augmented_ids.append(batch_ids)
            augmented_masks.append(batch_mask)
        
        # Pad to the same length
        max_length = max(ids.size(0) for ids in augmented_ids)
        padded_ids = []
        padded_masks = []
        
        for ids, mask in zip(augmented_ids, augmented_masks):
            pad_length = max_length - ids.size(0)
            if pad_length > 0:
                padded_ids.append(
                    F.pad(ids, (0, pad_length), value=self.pad_token_id)
                )
                padded_masks.append(
                    F.pad(mask, (0, pad_length), value=0)
                )
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)
        
        # Stack into tensors
        augmented_input_ids = torch.stack(padded_ids)
        augmented_attention_mask = torch.stack(padded_masks)
        
        # Forward through the language model
        outputs = self.language_model(
            input_ids=augmented_input_ids,
            attention_mask=augmented_attention_mask,
            labels=labels,
            return_dict=True
        )
        
        if return_dict:
            return self.ModelOutput(
                logits=outputs.logits,
                loss=outputs.loss,
                attention_weights=outputs.attention_weights,
                retrieved_docs=retrieved_indices
            )
        else:
            return outputs.logits, outputs.loss, outputs.attention_weights, retrieved_indices
    
    def get_token_efficiency(self) -> float:
        """
        Get the token efficiency ratio.
        
        Returns:
            efficiency: Token efficiency ratio (ratio of input tokens to total processed tokens)
        """
        # Traditional RAG is inefficient due to concatenation
        # For a rough estimate, assume original input is ~25% of total processed tokens
        return 0.25


class AttentionRAGModel(nn.Module):
    """
    Implementation of AttentionRAG: Attention-Guided Context Pruning in RAG.
    Based on: "AttentionRAG: Attention-Guided Context Pruning in Retrieval-Augmented Generation"
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_sequence_length: int = 4096,
        compression_ratio: float = 0.3,  # Target ratio of tokens to keep
        pad_token_id: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length
        self.compression_ratio = compression_ratio
        
        # Retriever component
        self.retriever = TraditionalRAGRetriever(embedding_dim, retrieval_count=5)
        
        # Query focus identification
        self.query_focus = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Context pruning attention
        self.pruning_attention = StandardTransformerAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Base language model (standard transformer)
        self.language_model = StandardTransformer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_sequence_length=max_sequence_length,
            pad_token_id=pad_token_id,
            dropout=dropout
        )
        
        # Document store (simulated)
        self.documents = []
        self.doc_token_ids = []
        
        # Create an output type for structured returns
        self.ModelOutput = namedtuple(
            "ModelOutput",
            ["logits", "loss", "attention_weights", "retrieved_docs", "pruning_ratio"]
        )
    
    def set_documents(
        self,
        documents: List[str],
        tokenizer
    ) -> None:
        """
        Set the document store and encode documents.
        
        Args:
            documents: List of document strings
            tokenizer: Tokenizer for encoding documents
        """
        self.documents = documents
        
        # Tokenize documents (simplified)
        self.doc_token_ids = []
        for doc in documents:
            # In a real implementation, this would be:
            # tokens = tokenizer.encode(doc, return_tensors="pt")
            # Here we just create random token IDs for simulation
            tokens = torch.randint(
                100, self.vocab_size, (1, min(100, self.max_sequence_length // 2))
            )
            self.doc_token_ids.append(tokens)
        
        # Encode documents for retrieval
        self.retriever.encode_documents(documents, tokenizer, self.language_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Tuple, namedtuple]:
        """
        Forward pass through the AttentionRAG model.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_length] containing token ids
            attention_mask: Optional tensor of shape [batch_size, seq_length] indicating
                           valid tokens (1) and padding (0)
            labels: Optional tensor of shape [batch_size, seq_length] containing target ids
            return_dict: Whether to return a namedtuple
        
        Returns:
            If return_dict=True, a namedtuple containing:
                - logits: Tensor of shape [batch_size, seq_length, vocab_size]
                - loss: Loss value if labels are provided, otherwise None
                - attention_weights: List of attention weight tensors
                - retrieved_docs: Indices of retrieved documents
                - pruning_ratio: Achieved pruning ratio
            
            If return_dict=False, a tuple of:
                (logits, loss, attention_weights, retrieved_docs, pruning_ratio)
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Get token embeddings from the language model
        token_embeds = self.language_model.token_embedding(input_ids)
        
        # Add positional embeddings
        positions = torch.arange(seq_length, device=device).expand(batch_size, seq_length)
        pos_embeds = self.language_model.pos_embedding(positions)
        
        embeddings = token_embeds + pos_embeds
        embeddings = self.language_model.dropout(embeddings)
        
        # Project to hidden dimension
        hidden_states = self.language_model.input_projection(embeddings)
        
        # Identify query focus
        focus_scores = self.query_focus(hidden_states).squeeze(-1)  # [batch_size, seq_length]
        focus_token_indices = torch.argmax(focus_scores, dim=-1)  # [batch_size]
        
        # Retrieve relevant documents
        retrieved_indices, _ = self.retriever(token_embeds)
        
        # Process retrieved documents and apply attention-guided pruning
        pruned_docs = []
        pruning_ratios = []
        
        for b in range(batch_size):
            # Get retrieved documents for this batch item
            batch_retrieved = []
            for idx in retrieved_indices[b]:
                idx = idx.item()
                if idx < len(self.doc_token_ids):
                    batch_retrieved.append(self.doc_token_ids[idx].to(device))
            
            if not batch_retrieved:
                pruned_docs.append(torch.tensor([], device=device))
                pruning_ratios.append(1.0)
                continue
            
            # Concatenate all retrieved documents
            concat_docs = torch.cat([doc.squeeze(0) for doc in batch_retrieved], dim=0)
            
            # Get embeddings for retrieved documents
            doc_embeds = self.language_model.token_embedding(concat_docs)
            doc_positions = torch.arange(concat_docs.size(0), device=device)
            doc_pos_embeds = self.language_model.pos_embedding(doc_positions)
            doc_embeddings = doc_embeds + doc_pos_embeds
            doc_hidden = self.language_model.input_projection(doc_embeddings)
            
            # Get query focus token
            focus_idx = focus_token_indices[b]
            focus_hidden = hidden_states[b, focus_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
            
            # Compute attention between focus token and documents
            _, attn_weights = self.pruning_attention(
                query=focus_hidden,
                key=doc_hidden.unsqueeze(0),
                value=doc_hidden.unsqueeze(0)
            )
            
            # Get important tokens based on attention weights
            importance = attn_weights[0, 0, 0]  # [doc_length]
            
            # Select top tokens based on compression ratio
            num_to_keep = max(1, int(self.compression_ratio * importance.size(0)))
            _, top_indices = torch.topk(importance, k=num_to_keep)
            
            # Sort indices to maintain document order
            top_indices, _ = torch.sort(top_indices)
            
            # Get pruned document tokens
            pruned_doc = concat_docs[top_indices]
            pruned_docs.append(pruned_doc)
            
            # Calculate achieved pruning ratio
            pruning_ratio = num_to_keep / importance.size(0)
            pruning_ratios.append(pruning_ratio)
        
        # Concatenate input with pruned documents
        augmented_ids = []
        augmented_masks = []
        
        for b in range(batch_size):
            pruned_doc = pruned_docs[b]
            
            if pruned_doc.numel() > 0:
                # Concatenate pruned document with input
                batch_ids = torch.cat([pruned_doc, input_ids[b]])
                batch_mask = torch.cat([
                    torch.ones(pruned_doc.size(0), device=device),
                    attention_mask[b]
                ])
                
                # Truncate if too long
                if batch_ids.size(0) > self.max_sequence_length:
                    batch_ids = batch_ids[-self.max_sequence_length:]
                    batch_mask = batch_mask[-self.max_sequence_length:]
            else:
                batch_ids = input_ids[b]
                batch_mask = attention_mask[b]
            
            augmented_ids.append(batch_ids)
            augmented_masks.append(batch_mask)
        
        # Pad to the same length
        max_length = max(ids.size(0) for ids in augmented_ids)
        padded_ids = []
        padded_masks = []
        
        for ids, mask in zip(augmented_ids, augmented_masks):
            pad_length = max_length - ids.size(0)
            if pad_length > 0:
                padded_ids.append(
                    F.pad(ids, (0, pad_length), value=self.pad_token_id)
                )
                padded_masks.append(
                    F.pad(mask, (0, pad_length), value=0)
                )
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)
        
        # Stack into tensors
        augmented_input_ids = torch.stack(padded_ids)
        augmented_attention_mask = torch.stack(padded_masks)
        
        # Forward through the language model
        outputs = self.language_model(
            input_ids=augmented_input_ids,
            attention_mask=augmented_attention_mask,
            labels=labels,
            return_dict=True
        )
        
        # Calculate average pruning ratio
        avg_pruning_ratio = sum(pruning_ratios) / len(pruning_ratios) if pruning_ratios else 1.0
        
        if return_dict:
            return self.ModelOutput(
                logits=outputs.logits,
                loss=outputs.loss,
                attention_weights=outputs.attention_weights,
                retrieved_docs=retrieved_indices,
                pruning_ratio=avg_pruning_ratio
            )
        else:
            return outputs.logits, outputs.loss, outputs.attention_weights, retrieved_indices, avg_pruning_ratio
    
    def get_token_efficiency(self) -> float:
        """
        Get the token efficiency ratio.
        
        Returns:
            efficiency: Token efficiency ratio (approximately the compression ratio)
        """
        return self.compression_ratio


class GCAModel(nn.Module):
    """
    Implementation of Grouped Cross Attention (GCA) for long-context language modeling.
    Based on: "Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling"
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_sequence_length: int = 4096,
        chunk_size: int = 128,
        top_k_chunks: int = 8,
        pad_token_id: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length
        self.chunk_size = chunk_size
        self.top_k_chunks = top_k_chunks
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        
        # Input projection
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Chunk encoder
        self.chunk_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Chunk query encoder
        self.query_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            StandardTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Create an output type for structured returns
        self.ModelOutput = namedtuple(
            "ModelOutput",
            ["logits", "loss", "attention_weights", "chunk_selection"]
        )
    
    def _split_into_chunks(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Split sequence into chunks.
        
        Args:
            x: Tensor of shape [batch_size, seq_length, hidden_dim]
            mask: Tensor of shape [batch_size, seq_length]
        
        Returns:
            chunks: Tensor of shape [batch_size, num_chunks, chunk_size, hidden_dim]
            chunk_mask: Tensor of shape [batch_size, num_chunks, chunk_size]
            chunk_lengths: List of actual chunk lengths
        """
        batch_size, seq_length, hidden_dim = x.shape
        
        # Calculate number of chunks
        num_chunks = (seq_length + self.chunk_size - 1) // self.chunk_size
        
        # Initialize output tensors
        chunks = torch.zeros(
            batch_size, num_chunks, self.chunk_size, hidden_dim, device=x.device
        )
        chunk_mask = torch.zeros(
            batch_size, num_chunks, self.chunk_size, device=mask.device
        )
        
        # Split sequence into chunks
        chunk_lengths = []
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, seq_length)
            actual_size = end_idx - start_idx
            chunk_lengths.append(actual_size)
            
            # Copy chunk
            chunks[:, i, :actual_size] = x[:, start_idx:end_idx]
            chunk_mask[:, i, :actual_size] = mask[:, start_idx:end_idx]
        
        return chunks, chunk_mask, chunk_lengths
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Tuple, namedtuple]:
        """
        Forward pass through the GCA model.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_length] containing token ids
            attention_mask: Optional tensor of shape [batch_size, seq_length] indicating
                           valid tokens (1) and padding (0)
            labels: Optional tensor of shape [batch_size, seq_length] containing target ids
            return_dict: Whether to return a namedtuple
        
        Returns:
            If return_dict=True, a namedtuple containing:
                - logits: Tensor of shape [batch_size, seq_length, vocab_size]
                - loss: Loss value if labels are provided, otherwise None
                - attention_weights: List of attention weight tensors
                - chunk_selection: Tensor indicating which chunks were selected
            
            If return_dict=False, a tuple of:
                (logits, loss, attention_weights, chunk_selection)
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Add positional embeddings
        positions = torch.arange(seq_length, device=device).expand(batch_size, seq_length)
        pos_embeds = self.pos_embedding(positions)
        
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        
        # Project embeddings to hidden dimension
        hidden_states = self.input_projection(embeddings)
        
        # Split into chunks
        chunks, chunk_mask, chunk_lengths = self._split_into_chunks(hidden_states, attention_mask)
        
        # Encode chunks
        batch_size, num_chunks, chunk_size, hidden_dim = chunks.shape
        
        # Average pool each chunk
        chunk_mask_expanded = chunk_mask.unsqueeze(-1)
        masked_chunks = chunks * chunk_mask_expanded
        
        # Sum and divide by non-zero elements
        chunk_sum = torch.sum(masked_chunks, dim=2)
        mask_sum = torch.sum(chunk_mask_expanded, dim=2) + 1e-10  # Avoid division by zero
        chunk_avg = chunk_sum / mask_sum
        
        # Encode chunk representations
        chunk_encodings = self.chunk_encoder(chunk_avg)  # [batch_size, num_chunks, hidden_dim]
        
        # Process through transformer layers with top-k chunk selection
        all_attention_weights = []
        all_chunk_selections = []
        
        for layer_idx, layer in enumerate(self.layers):
            # For each token, find the most relevant chunks
            query_encodings = self.query_encoder(hidden_states)  # [batch_size, seq_length, hidden_dim]
            
            # Compute relevance scores between queries and chunks
            # [batch_size, seq_length, num_chunks]
            chunk_scores = torch.bmm(
                query_encodings,
                chunk_encodings.transpose(1, 2)
            )
            
            # Select top-k chunks for each token
            # [batch_size, seq_length, top_k_chunks]
            _, top_k_indices = torch.topk(
                chunk_scores, k=min(self.top_k_chunks, num_chunks), dim=-1
            )
            
            # Create selection mask
            # [batch_size, seq_length, num_chunks]
            chunk_selection = torch.zeros(
                batch_size, seq_length, num_chunks, device=device
            )
            
            # Set selected chunks to 1
            for b in range(batch_size):
                for s in range(seq_length):
                    chunk_selection[b, s, top_k_indices[b, s]] = 1
            
            all_chunk_selections.append(chunk_selection)
            
            # For each token, gather selected chunks
            selected_states = []
            
            for b in range(batch_size):
                token_states = []
                
                for s in range(seq_length):
                    # Get selected chunks for this token
                    selected_indices = top_k_indices[b, s]
                    
                    # Gather tokens from selected chunks
                    gathered_tokens = []
                    for idx in selected_indices:
                        idx = idx.item()
                        if idx < num_chunks:
                            # Get all tokens from this chunk
                            chunk_tokens = chunks[b, idx]
                            
                            # Only include valid tokens (not padding)
                            valid_mask = chunk_mask[b, idx].bool()
                            
                            if valid_mask.any():
                                gathered_tokens.append(chunk_tokens[valid_mask])
                    
                    if gathered_tokens:
                        # Concatenate gathered tokens
                        gathered = torch.cat(gathered_tokens, dim=0)
                        
                        # Use gathered tokens for this position
                        gathered_context = torch.cat([
                            gathered,
                            hidden_states[b, s].unsqueeze(0)  # Add current token
                        ], dim=0)
                    else:
                        # Just use current token if no chunks were selected
                        gathered_context = hidden_states[b, s].unsqueeze(0)
                    
                    token_states.append(gathered_context)
                
                selected_states.append(token_states)
            
            # Process each token with local attention
            new_states = torch.zeros_like(hidden_states)
            
            for b in range(batch_size):
                for s in range(seq_length):
                    context = selected_states[b][s]
                    context_length = context.size(0)
                    
                    # Create self-attention input
                    # Last token is the query token
                    query = context[-1:].unsqueeze(0)  # [1, 1, hidden_dim]
                    
                    # Use all tokens as keys and values
                    key_value = context.unsqueeze(0)  # [1, context_length, hidden_dim]
                    
                    # Apply self-attention
                    attn_output, attn_weights = layer.attention(
                        query=query,
                        key=key_value,
                        value=key_value
                    )
                    
                    # Store attention weights
                    if layer_idx == 0 and s == 0:
                        all_attention_weights.append(attn_weights)
                    
                    # Apply feed-forward
                    output, _ = layer.ffn(attn_output)
                    
                    # Store result
                    new_states[b, s] = output[0, 0]
            
            # Update hidden states
            hidden_states = new_states
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        if return_dict:
            return self.ModelOutput(
                logits=logits,
                loss=loss,
                attention_weights=all_attention_weights,
                chunk_selection=all_chunk_selections
            )
        else:
            return logits, loss, all_attention_weights, all_chunk_selections
    
    def get_token_efficiency(self) -> float:
        """
        Get the token efficiency ratio.
        
        Returns:
            efficiency: Token efficiency ratio (top-k/total chunks)
        """
        return self.top_k_chunks / (self.max_sequence_length / self.chunk_size)


class RazorAttentionModel(nn.Module):
    """
    Implementation of RazorAttention: Efficient KV Cache Compression Through Retrieval Heads.
    Based on: "RazorAttention: Efficient KV Cache Compression Through Retrieval Heads"
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_sequence_length: int = 4096,
        retrieval_head_ratio: float = 0.25,  # Ratio of heads to use as retrieval heads
        compression_ratio: float = 0.3,  # Ratio of tokens to keep in non-retrieval heads
        pad_token_id: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length
        
        # Calculate number of retrieval heads
        self.num_retrieval_heads = max(1, int(num_heads * retrieval_head_ratio))
        self.compression_ratio = compression_ratio
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        
        # Input projection
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Modified transformer attention with retrieval heads
        class RazorTransformerLayer(StandardTransformerLayer):
            def __init__(
                self_layer,
                hidden_dim: int,
                num_heads: int,
                num_retrieval_heads: int,
                compression_ratio: float,
                dropout: float = 0.1
            ):
                super().__init__(hidden_dim, num_heads, dropout)
                self_layer.num_retrieval_heads = num_retrieval_heads
                self_layer.compression_ratio = compression_ratio
                
                # Register retrieval head indices
                retrieval_head_indices = torch.randperm(num_heads)[:num_retrieval_heads]
                self_layer.register_buffer("retrieval_head_indices", retrieval_head_indices)
                
                # Compensation token
                self_layer.compensation_token = nn.Parameter(torch.randn(hidden_dim // num_heads))
            
            def forward(
                self_layer,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                """
                Forward pass with RazorAttention.
                """
                batch_size, seq_length, _ = x.shape
                
                # Self-attention
                residual = x
                x = self_layer.norm1(x)
                
                # Project query, key, value
                q = self_layer.attention.q_proj(x)
                k = self_layer.attention.k_proj(x)
                v = self_layer.attention.v_proj(x)
                
                # Reshape for multi-head attention
                head_dim = self_layer.attention.head_dim
                q = q.view(batch_size, seq_length, self_layer.attention.num_heads, head_dim)
                k = k.view(batch_size, seq_length, self_layer.attention.num_heads, head_dim)
                v = v.view(batch_size, seq_length, self_layer.attention.num_heads, head_dim)
                
                # Transpose to [batch_size, num_heads, seq_length, head_dim]
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                
                # Apply RazorAttention
                outputs = []
                attention_weights = []
                
                for h in range(self_layer.attention.num_heads):
                    if h in self_layer.retrieval_head_indices:
                        # Retrieval head - use full attention
                        # [batch_size, seq_length, seq_length]
                        attn_scores = torch.bmm(
                            q[:, h],  # [batch_size, seq_length, head_dim]
                            k[:, h].transpose(1, 2)  # [batch_size, head_dim, seq_length]
                        ) * self_layer.attention.scale
                        
                        # Apply attention mask if provided
                        if attention_mask is not None:
                            attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))
                        
                        # Apply softmax and dropout
                        attn_weights_h = F.softmax(attn_scores, dim=-1)
                        attn_weights_h = self_layer.attention.dropout(attn_weights_h)
                        
                        # Apply attention weights to values
                        # [batch_size, seq_length, head_dim]
                        output_h = torch.bmm(attn_weights_h, v[:, h])
                        
                        outputs.append(output_h)
                        attention_weights.append(attn_weights_h)
                    else:
                        # Non-retrieval head - compress KV cache
                        # Compute attention scores for all tokens
                        # [batch_size, seq_length, seq_length]
                        attn_scores = torch.bmm(
                            q[:, h],  # [batch_size, seq_length, head_dim]
                            k[:, h].transpose(1, 2)  # [batch_size, head_dim, seq_length]
                        ) * self_layer.attention.scale
                        
                        # Apply attention mask if provided
                        if attention_mask is not None:
                            attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))
                        
                        # Find most important tokens based on max attention score
                        # [batch_size, seq_length]
                        importance = torch.max(attn_scores, dim=1)[0]
                        
                        # Select top tokens based on compression ratio
                        num_to_keep = max(1, int(self_layer.compression_ratio * seq_length))
                        
                        # [batch_size, num_to_keep]
                        _, top_indices = torch.topk(importance, k=num_to_keep, dim=-1)
                        
                        # Create selection mask
                        # [batch_size, seq_length]
                        selection_mask = torch.zeros(
                            batch_size, seq_length, device=x.device, dtype=torch.bool
                        )
                        
                        for b in range(batch_size):
                            selection_mask[b, top_indices[b]] = True
                        
                        # Add compensation token
                        # Create compensation token from discarded tokens
                        comp_tokens = []
                        
                        for b in range(batch_size):
                            # Get discarded tokens
                            discard_mask = ~selection_mask[b]
                            
                            if discard_mask.any():
                                # Average discarded tokens
                                discarded_k = k[b, h, discard_mask]
                                discarded_v = v[b, h, discard_mask]
                                
                                # Use weighted average based on importance
                                importance_weights = importance[b, discard_mask]
                                importance_weights = F.softmax(importance_weights, dim=0)
                                
                                comp_k = torch.sum(discarded_k * importance_weights.unsqueeze(-1), dim=0)
                                comp_v = torch.sum(discarded_v * importance_weights.unsqueeze(-1), dim=0)
                            else:
                                # No tokens discarded, use learnable compensation token
                                comp_k = self_layer.compensation_token
                                comp_v = self_layer.compensation_token
                            
                            comp_tokens.append((comp_k, comp_v))
                        
                        # Compute attention with selected tokens and compensation token
                        outputs_h = []
                        attn_weights_h = []
                        
                        for b in range(batch_size):
                            # Get selected tokens
                            selected_k = k[b, h, selection_mask[b]]  # [selected_length, head_dim]
                            selected_v = v[b, h, selection_mask[b]]  # [selected_length, head_dim]
                            
                            # Add compensation token
                            comp_k, comp_v = comp_tokens[b]
                            selected_k = torch.cat([selected_k, comp_k.unsqueeze(0)], dim=0)
                            selected_v = torch.cat([selected_v, comp_v.unsqueeze(0)], dim=0)
                            
                            # For each query token
                            query_outputs = []
                            query_weights = []
                            
                            for s in range(seq_length):
                                # Get query token
                                query = q[b, h, s].unsqueeze(0)  # [1, head_dim]
                                
                                # Compute attention with selected tokens
                                # [1, selected_length + 1]
                                scores = torch.mm(
                                    query,
                                    selected_k.transpose(0, 1)
                                ) * self_layer.attention.scale
                                
                                # Apply softmax and dropout
                                weights = F.softmax(scores, dim=-1)
                                weights = self_layer.attention.dropout(weights)
                                
                                # Apply attention weights to values
                                # [1, head_dim]
                                output = torch.mm(weights, selected_v)
                                
                                query_outputs.append(output)
                                query_weights.append(weights)
                            
                            # Stack query outputs and weights
                            # [seq_length, head_dim]
                            outputs_h.append(torch.cat(query_outputs, dim=0))
                            
                            # Create full attention weight matrix (for visualization)
                            full_weights = torch.zeros(
                                seq_length, seq_length, device=x.device
                            )
                            
                            for s in range(seq_length):
                                # Copy weights to selected positions
                                full_weights[s, selection_mask[b]] = query_weights[s][0, :-1]
                                
                                # Last weight is for compensation token
                                # Distribute it evenly among discarded tokens
                                discard_mask = ~selection_mask[b]
                                if discard_mask.any():
                                    num_discarded = discard_mask.sum().item()
                                    comp_weight = query_weights[s][0, -1].item() / num_discarded
                                    full_weights[s, discard_mask] = comp_weight
                            
                            attn_weights_h.append(full_weights)
                        
                        # Stack batch outputs
                        # [batch_size, seq_length, head_dim]
                        outputs.append(torch.stack(outputs_h))
                        
                        # Stack batch weights
                        # [batch_size, seq_length, seq_length]
                        attention_weights.append(torch.stack(attn_weights_h))
                
                # Concatenate outputs from all heads
                # [batch_size, seq_length, hidden_dim]
                output = torch.cat([output.unsqueeze(1) for output in outputs], dim=1)
                output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
                
                # Stack attention weights from all heads
                # [batch_size, num_heads, seq_length, seq_length]
                stacked_weights = torch.stack(attention_weights, dim=1)
                
                # Apply output projection
                output = self_layer.attention.out_proj(output)
                
                # Residual connection
                x = residual + self_layer.dropout(output)
                
                # Feed-forward network
                residual = x
                x = self_layer.norm2(x)
                x = residual + self_layer.dropout(self_layer.ffn(x))
                
                return x, {'attention_weights': stacked_weights}
        
        # Transformer layers with RazorAttention
        self.layers = nn.ModuleList([
            RazorTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_retrieval_heads=self.num_retrieval_heads,
                compression_ratio=self.compression_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Create an output type for structured returns
        self.ModelOutput = namedtuple(
            "ModelOutput",
            ["logits", "loss", "attention_weights"]
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Tuple, namedtuple]:
        """
        Forward pass through the RazorAttention model.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_length] containing token ids
            attention_mask: Optional tensor of shape [batch_size, seq_length] indicating
                           valid tokens (1) and padding (0)
            labels: Optional tensor of shape [batch_size, seq_length] containing target ids
            return_dict: Whether to return a namedtuple
        
        Returns:
            If return_dict=True, a namedtuple containing:
                - logits: Tensor of shape [batch_size, seq_length, vocab_size]
                - loss: Loss value if labels are provided, otherwise None
                - attention_weights: List of attention weight tensors
            
            If return_dict=False, a tuple of:
                (logits, loss, attention_weights)
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Add positional embeddings
        positions = torch.arange(seq_length, device=device).expand(batch_size, seq_length)
        pos_embeds = self.pos_embedding(positions)
        
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        
        # Project embeddings to hidden dimension
        hidden_states = self.input_projection(embeddings)
        
        # Prepare attention mask for self-attention
        attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
        attn_mask = attn_mask.expand(-1, -1, seq_length, -1)  # [batch_size, 1, seq_length, seq_length]
        
        # Process through transformer layers
        all_attention_weights = []
        
        for layer in self.layers:
            hidden_states, layer_outputs = layer(hidden_states, attention_mask=attn_mask)
            all_attention_weights.append(layer_outputs['attention_weights'])
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        if return_dict:
            return self.ModelOutput(
                logits=logits,
                loss=loss,
                attention_weights=all_attention_weights
            )
        else:
            return logits, loss, all_attention_weights
    
    def get_token_efficiency(self) -> float:
        """
        Get the token efficiency ratio.
        
        Returns:
            efficiency: Token efficiency ratio
        """
        # Full KV cache for retrieval heads, compressed for non-retrieval heads
        retrieval_ratio = self.num_retrieval_heads / self.num_heads
        compressed_ratio = (1 - retrieval_ratio) * self.compression_ratio
        
        return retrieval_ratio + compressed_ratio


class PyramidKVModel(nn.Module):
    """
    Implementation of PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling.
    Based on: "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling"
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        max_sequence_length: int = 4096,
        compression_schedule: Optional[List[float]] = None,
        pad_token_id: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length
        
        # Define layer-wise compression schedule
        # Higher layers get more compression
        if compression_schedule is None:
            self.compression_schedule = [
                1.0 - (i / (num_layers - 1)) * 0.8 for i in range(num_layers)
            ]
        else:
            self.compression_schedule = compression_schedule
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        
        # Input projection
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Modified transformer layers with dynamic KV cache compression
        class PyramidKVLayer(StandardTransformerLayer):
            def __init__(
                self_layer,
                hidden_dim: int,
                num_heads: int,
                compression_ratio: float,
                dropout: float = 0.1
            ):
                super().__init__(hidden_dim, num_heads, dropout)
                self_layer.compression_ratio = compression_ratio
            
            def forward(
                self_layer,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                """
                Forward pass with PyramidKV.
                """
                batch_size, seq_length, _ = x.shape
                
                # Self-attention
                residual = x
                x = self_layer.norm1(x)
                
                # Project query, key, value
                q = self_layer.attention.q_proj(x)
                k = self_layer.attention.k_proj(x)
                v = self_layer.attention.v_proj(x)
                
                # Reshape for multi-head attention
                head_dim = self_layer.attention.head_dim
                q = q.view(batch_size, seq_length, self_layer.attention.num_heads, head_dim)
                k = k.view(batch_size, seq_length, self_layer.attention.num_heads, head_dim)
                v = v.view(batch_size, seq_length, self_layer.attention.num_heads, head_dim)
                
                # Transpose to [batch_size, num_heads, seq_length, head_dim]
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                
                # Apply compression based on layer's compression ratio
                if seq_length > 1 and self_layer.compression_ratio < 1.0:
                    # Compute attention scores for compression
                    # [batch_size, num_heads, seq_length, seq_length]
                    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self_layer.attention.scale
                    
                    # Apply attention mask if provided
                    if attention_mask is not None:
                        # Expand the mask for all heads
                        expanded_mask = attention_mask.expand(-1, self_layer.attention.num_heads, -1, -1)
                        attn_scores = attn_scores.masked_fill(~expanded_mask.bool(), float('-inf'))
                    
                    # Calculate token importance based on attention
                    # [batch_size, num_heads, seq_length]
                    importance = torch.sum(torch.abs(attn_scores), dim=2)
                    
                    # Average across heads
                    # [batch_size, seq_length]
                    avg_importance = torch.mean(importance, dim=1)
                    
                    # Number of tokens to keep
                    num_to_keep = max(1, int(self_layer.compression_ratio * seq_length))
                    
                    # Select top tokens
                    # [batch_size, num_to_keep]
                    _, top_indices = torch.topk(avg_importance, k=num_to_keep, dim=-1)
                    
                    # Create compressed key and value tensors
                    compressed_k = torch.zeros(
                        batch_size, self_layer.attention.num_heads, num_to_keep, head_dim,
                        device=x.device
                    )
                    
                    compressed_v = torch.zeros(
                        batch_size, self_layer.attention.num_heads, num_to_keep, head_dim,
                        device=x.device
                    )
                    
                    # Fill with selected tokens
                    for b in range(batch_size):
                        compressed_k[b, :, :, :] = k[b, :, top_indices[b], :]
                        compressed_v[b, :, :, :] = v[b, :, top_indices[b], :]
                    
                    # Create compressed attention mask
                    if attention_mask is not None:
                        compressed_mask = torch.zeros(
                            batch_size, 1, seq_length, num_to_keep,
                            device=x.device
                        )
                        
                        for b in range(batch_size):
                            compressed_mask[b, :, :, :] = attention_mask[b, :, :, top_indices[b]]
                    else:
                        compressed_mask = None
                    
                    # Use compressed keys and values
                    k_for_attn = compressed_k
                    v_for_attn = compressed_v
                    mask_for_attn = compressed_mask
                else:
                    # No compression
                    k_for_attn = k
                    v_for_attn = v
                    mask_for_attn = attention_mask
                
                # Compute attention with compressed KV
                # [batch_size, num_heads, seq_length, compressed_length]
                compressed_length = k_for_attn.size(2)
                attn_scores = torch.matmul(q, k_for_attn.transpose(-2, -1)) * self_layer.attention.scale
                
                # Apply attention mask if provided
                if mask_for_attn is not None:
                    attn_scores = attn_scores.masked_fill(~mask_for_attn.bool(), float('-inf'))
                
                # Apply softmax and dropout
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self_layer.attention.dropout(attn_weights)
                
                # Apply attention weights to values
                # [batch_size, num_heads, seq_length, head_dim]
                attn_output = torch.matmul(attn_weights, v_for_attn)
                
                # Transpose and reshape
                # [batch_size, seq_length, num_heads, head_dim]
                attn_output = attn_output.transpose(1, 2).contiguous()
                
                # [batch_size, seq_length, hidden_dim]
                attn_output = attn_output.view(batch_size, seq_length, self_layer.hidden_dim)
                
                # Apply output projection
                output = self_layer.attention.out_proj(attn_output)
                
                # Residual connection
                x = residual + self_layer.dropout(output)
                
                # Feed-forward network
                residual = x
                x = self_layer.norm2(x)
                x = residual + self_layer.dropout(self_layer.ffn(x))
                
                # For attention visualization
                # Create full attention weights matrix
                full_attn_weights = torch.zeros(
                    batch_size, self_layer.attention.num_heads, seq_length, seq_length,
                    device=x.device
                )
                
                if seq_length > 1 and self_layer.compression_ratio < 1.0:
                    for b in range(batch_size):
                        full_attn_weights[b, :, :, top_indices[b]] = attn_weights[b]
                else:
                    full_attn_weights = attn_weights
                
                return x, {'attention_weights': full_attn_weights, 'compression_ratio': self_layer.compression_ratio}
        
        # Transformer layers with PyramidKV
        self.layers = nn.ModuleList([
            PyramidKVLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                compression_ratio=self.compression_schedule[i],
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Create an output type for structured returns
        self.ModelOutput = namedtuple(
            "ModelOutput",
            ["logits", "loss", "attention_weights", "compression_ratios"]
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Tuple, namedtuple]:
        """
        Forward pass through the PyramidKV model.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_length] containing token ids
            attention_mask: Optional tensor of shape [batch_size, seq_length] indicating
                           valid tokens (1) and padding (0)
            labels: Optional tensor of shape [batch_size, seq_length] containing target ids
            return_dict: Whether to return a namedtuple
        
        Returns:
            If return_dict=True, a namedtuple containing:
                - logits: Tensor of shape [batch_size, seq_length, vocab_size]
                - loss: Loss value if labels are provided, otherwise None
                - attention_weights: List of attention weight tensors
                - compression_ratios: List of compression ratios per layer
            
            If return_dict=False, a tuple of:
                (logits, loss, attention_weights, compression_ratios)
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).float()
        
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Add positional embeddings
        positions = torch.arange(seq_length, device=device).expand(batch_size, seq_length)
        pos_embeds = self.pos_embedding(positions)
        
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        
        # Project embeddings to hidden dimension
        hidden_states = self.input_projection(embeddings)
        
        # Prepare attention mask for self-attention
        attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
        attn_mask = attn_mask.expand(-1, -1, seq_length, -1)  # [batch_size, 1, seq_length, seq_length]
        
        # Process through transformer layers
        all_attention_weights = []
        compression_ratios = []
        
        for layer in self.layers:
            hidden_states, layer_outputs = layer(hidden_states, attention_mask=attn_mask)
            all_attention_weights.append(layer_outputs['attention_weights'])
            compression_ratios.append(layer_outputs['compression_ratio'])
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        if return_dict:
            return self.ModelOutput(
                logits=logits,
                loss=loss,
                attention_weights=all_attention_weights,
                compression_ratios=compression_ratios
            )
        else:
            return logits, loss, all_attention_weights, compression_ratios
    
    def get_token_efficiency(self) -> float:
        """
        Get the token efficiency ratio.
        
        Returns:
            efficiency: Token efficiency ratio (average compression across layers)
        """
        return sum(self.compression_schedule) / len(self.compression_schedule)