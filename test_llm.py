"""
Sparse Mixture-of-Experts (MoE) Transformer Implementation
Based on Gemini 2.5 architecture description

This implementation provides a sparse MoE architecture that activates only a subset
of expert parameters per input token, allowing for decoupling of model capacity
from computation cost.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor


class Expert(nn.Module):
    """
    Individual expert network in the MoE architecture.

    Each expert is a feed-forward network that specializes in processing
    certain types of input patterns.

    Args:
        hidden_dim: Hidden dimension size
        intermediate_dim: Intermediate dimension in feed-forward network
        dropout: Dropout probability
        activation: Activation function to use
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        dropout: float = 0.1,
        activation: str = "swish",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        # Feed-forward network
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with proper scaling."""
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the expert network.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_dim]
        """
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class Router(nn.Module):
    """
    Gating network that routes tokens to appropriate experts.

    The router learns to assign input tokens to the most suitable experts
    based on the token representations.

    Args:
        hidden_dim: Hidden dimension size
        num_experts: Number of experts in the MoE layer
        top_k: Number of experts to activate per token
        temperature: Temperature for softmax routing
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature

        # Linear layer for routing scores
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize routing weights."""
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Route tokens to experts.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple of (routing_weights, expert_indices, routing_probs)
            - routing_weights: [batch_size, seq_len, top_k]
            - expert_indices: [batch_size, seq_len, top_k]
            - routing_probs: [batch_size, seq_len, num_experts]
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Compute routing scores
        routing_logits = self.gate(
            x
        )  # [batch_size, seq_len, num_experts]
        routing_logits = routing_logits / self.temperature

        # Apply softmax to get probabilities
        routing_probs = F.softmax(routing_logits, dim=-1)

        # Select top-k experts
        routing_weights, expert_indices = torch.topk(
            routing_probs, self.top_k, dim=-1
        )

        # Normalize routing weights
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )

        return routing_weights, expert_indices, routing_probs


class MoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer.

    This layer contains multiple expert networks and a router that decides
    which experts to activate for each input token.

    Args:
        hidden_dim: Hidden dimension size
        num_experts: Number of expert networks
        top_k: Number of experts to activate per token
        intermediate_dim: Intermediate dimension in expert networks
        dropout: Dropout probability
        activation: Activation function for experts
        load_balance_weight: Weight for load balancing loss
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "swish",
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 4

        # Create expert networks
        self.experts = nn.ModuleList(
            [
                Expert(
                    hidden_dim, intermediate_dim, dropout, activation
                )
                for _ in range(num_experts)
            ]
        )

        # Router for expert selection
        self.router = Router(hidden_dim, num_experts, top_k)

        logger.info(
            f"Created MoE layer with {num_experts} experts, top_k={top_k}"
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple of (output, aux_losses)
            - output: [batch_size, seq_len, hidden_dim]
            - aux_losses: Dictionary containing auxiliary losses
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Get routing decisions
        routing_weights, expert_indices, routing_probs = self.router(
            x
        )

        # Initialize output
        output = torch.zeros_like(x)

        # Process each expert
        for i in range(self.num_experts):
            # Create mask for tokens routed to this expert
            expert_mask = (expert_indices == i).any(
                dim=-1
            )  # [batch_size, seq_len]

            if not expert_mask.any():
                continue

            # Get tokens for this expert
            expert_tokens = x[expert_mask]  # [num_tokens, hidden_dim]

            if expert_tokens.numel() == 0:
                continue

            # Process through expert
            expert_output = self.experts[i](expert_tokens)

            # Compute weights for this expert
            expert_weights = torch.zeros(
                batch_size, seq_len, device=x.device
            )
            for k in range(self.top_k):
                mask = expert_indices[:, :, k] == i
                expert_weights[mask] = routing_weights[:, :, k][mask]

            # Add weighted expert output
            expert_contribution = torch.zeros_like(x)
            expert_contribution[expert_mask] = expert_output
            output += expert_contribution * expert_weights.unsqueeze(
                -1
            )

        # Compute auxiliary losses
        aux_losses = self._compute_aux_losses(
            routing_probs, expert_indices
        )

        return output, aux_losses

    def _compute_aux_losses(
        self, routing_probs: Tensor, expert_indices: Tensor
    ) -> Dict[str, Tensor]:
        """
        Compute auxiliary losses for training stability.

        Args:
            routing_probs: Routing probabilities [batch_size, seq_len, num_experts]
            expert_indices: Selected expert indices [batch_size, seq_len, top_k]

        Returns:
            Dictionary of auxiliary losses
        """
        batch_size, seq_len, num_experts = routing_probs.shape

        # Load balancing loss
        expert_usage = torch.zeros(
            num_experts, device=routing_probs.device
        )
        total_tokens = batch_size * seq_len * self.top_k

        for i in range(num_experts):
            expert_usage[i] = (
                expert_indices == i
            ).sum().float() / total_tokens

        target_usage = 1.0 / num_experts
        load_balance_loss = F.mse_loss(
            expert_usage, torch.full_like(expert_usage, target_usage)
        )

        # Entropy loss to encourage diversity
        entropy_loss = (
            -(routing_probs * torch.log(routing_probs + 1e-8))
            .sum(dim=-1)
            .mean()
        )

        return {
            "load_balance_loss": load_balance_loss
            * self.load_balance_weight,
            "entropy_loss": entropy_loss * 0.01,
            "expert_usage": expert_usage,
        }


class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE feed-forward layer.

    This block combines multi-head attention with a sparse MoE layer,
    following the standard transformer architecture pattern.

    Args:
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        num_experts: Number of experts in MoE layer
        top_k: Number of experts to activate per token
        dropout: Dropout probability
        layer_norm_eps: Epsilon for layer normalization
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # MoE layer
        self.moe_layer = MoELayer(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask

        Returns:
            Tuple of (output, aux_losses)
        """
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attention(
            x, x, x, key_padding_mask=attention_mask
        )
        x = residual + self.dropout(attn_output)

        # MoE layer with residual connection
        residual = x
        x = self.norm2(x)
        moe_output, aux_losses = self.moe_layer(x)
        x = residual + self.dropout(moe_output)

        return x, aux_losses


class MoETransformer(nn.Module):
    """
    Complete sparse MoE Transformer model.

    This model implements the full transformer architecture with sparse
    mixture-of-experts layers, similar to the Gemini 2.5 architecture.

    Args:
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_experts: Number of experts per MoE layer
        top_k: Number of experts to activate per token
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_experts: int,
        top_k: int = 2,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                MoETransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_experts=num_experts,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_projection = nn.Linear(
            hidden_dim, vocab_size, bias=False
        )

        # Tie input and output embeddings
        self.output_projection.weight = self.token_embedding.weight

        self._init_weights()

        logger.info(
            f"Created MoE Transformer with {num_layers} layers, "
            f"{num_experts} experts per layer, hidden_dim={hidden_dim}"
        )

    def _init_weights(self) -> None:
        """Initialize model weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Initialize output projection
        nn.init.normal_(self.output_projection.weight, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_aux_losses: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            return_aux_losses: Whether to return auxiliary losses

        Returns:
            If return_aux_losses=False: logits [batch_size, seq_len, vocab_size]
            If return_aux_losses=True: (logits, aux_losses)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]

        # Collect auxiliary losses
        all_aux_losses = {}

        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            x, aux_losses = layer(x, attention_mask)

            if return_aux_losses:
                for key, value in aux_losses.items():
                    if key not in all_aux_losses:
                        all_aux_losses[key] = []
                    all_aux_losses[key].append(value)

        # Final layer norm
        x = self.final_norm(x)

        # Output projection
        logits = self.output_projection(x)

        if not return_aux_losses:
            return logits

        # Average auxiliary losses across layers
        avg_aux_losses = {}
        for key, values in all_aux_losses.items():
            if key == "expert_usage":
                # For expert usage, we want to see all layers
                avg_aux_losses[key] = torch.stack(values)
            else:
                avg_aux_losses[key] = torch.stack(values).mean()

        return logits, avg_aux_losses

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_active_parameters(self) -> int:
        """Get number of active parameters per forward pass."""
        # This is approximate - actual active parameters depend on routing
        total_params = self.get_num_parameters()

        # Estimate active expert parameters
        expert_params_per_layer = 0
        for layer in self.layers:
            expert_params = sum(
                p.numel()
                for p in layer.moe_layer.experts[0].parameters()
            )
            expert_params_per_layer += (
                expert_params * layer.moe_layer.top_k
            )

        total_expert_params = sum(
            sum(
                p.numel()
                for expert in layer.moe_layer.experts
                for p in expert.parameters()
            )
            for layer in self.layers
        )

        active_params = (
            total_params
            - total_expert_params
            + expert_params_per_layer * len(self.layers)
        )
        return active_params


# Example usage and testing
if __name__ == "__main__":
    # Configure logger
    logger.add("moe_training.log", rotation="500 MB", level="INFO")

    # Model configuration
    config = {
        "vocab_size": 32000,
        "hidden_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "num_experts": 8,
        "top_k": 2,
        "max_seq_len": 2048,
        "dropout": 0.1,
    }

    # Create model
    model = MoETransformer(**config)

    # Print model info
    total_params = model.get_num_parameters()
    active_params = model.get_num_active_parameters()

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(
        f"Active parameters per forward pass: {active_params:,}"
    )
    logger.info(
        f"Parameter efficiency: {active_params/total_params:.2%}"
    )

    # Test forward pass
    batch_size, seq_len = 2, 512
    input_ids = torch.randint(
        0, config["vocab_size"], (batch_size, seq_len)
    )

    with torch.no_grad():
        logits, aux_losses = model(input_ids)

    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Output shape: {logits.shape}")
    logger.info(f"Auxiliary losses: {list(aux_losses.keys())}")

    # Print expert usage statistics
    expert_usage = aux_losses[
        "expert_usage"
    ]  # [num_layers, num_experts]
    logger.info(f"Expert usage shape: {expert_usage.shape}")
    logger.info(f"Average expert usage: {expert_usage.mean(dim=0)}")
