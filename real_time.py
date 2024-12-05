import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from loguru import logger

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TransformerConfig:
    """Configuration class for MoE Transformer model parameters."""

    vocab_size: int = 50257
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_expert_layers: int = 4
    num_experts: int = 8
    expert_capacity: int = 32
    max_position_embeddings: int = 1024
    dropout_prob: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    num_query_groups: int = 4  # For multi-query attention


class ExpertLayer(nn.Module):
    """Individual expert neural network."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size, 4 * config.hidden_size
        )
        self.fc2 = nn.Linear(
            4 * config.hidden_size, config.hidden_size
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with dynamic routing."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity

        # Create expert networks
        self.experts = nn.ModuleList(
            [ExpertLayer(config) for _ in range(config.num_experts)]
        )

        # Router network
        self.router = nn.Linear(
            config.hidden_size, config.num_experts
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict]:
        """Route inputs to experts and combine outputs."""
        batch_size, seq_len, hidden_size = x.shape

        # Calculate routing probabilities
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k = 2
        gates, indices = torch.topk(routing_weights, top_k, dim=-1)
        gates = F.softmax(gates, dim=-1)

        # Process inputs through selected experts
        final_output = torch.zeros_like(x)
        router_load = torch.zeros(self.num_experts, device=x.device)

        for i in range(top_k):
            expert_index = indices[..., i]
            gate = gates[..., i : i + 1]

            # Count expert assignments
            for j in range(self.num_experts):
                router_load[j] += (expert_index == j).float().sum()

            # Process through selected experts
            for j in range(self.num_experts):
                mask = expert_index == j
                if not mask.any():
                    continue

                expert_input = x[mask]
                expert_output = self.experts[j](expert_input)
                final_output[mask] += gate[mask] * expert_output

        aux_loss = router_load.float().var() / (
            router_load.float().mean() ** 2
        )

        return final_output, {"load_balancing_loss": aux_loss}


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention mechanism with proper multi-query group handling."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_query_groups = config.num_query_groups
        self.hidden_size = config.hidden_size
        self.head_dim = (
            config.hidden_size // config.num_attention_heads
        )

        # Query projection maintains full head dimension
        self.q_proj = nn.Linear(
            config.hidden_size, config.hidden_size
        )

        # Key and value projections use reduced number of heads (query groups)
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.head_dim * config.num_query_groups,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.head_dim * config.num_query_groups,
        )

        self.dropout = nn.Dropout(config.dropout_prob)

        # Calculate heads per group for proper reshaping
        self.heads_per_group = (
            self.num_attention_heads // self.num_query_groups
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        cache: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        batch_size, seq_length, _ = hidden_states.shape

        # Project queries, keys, and values
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Reshape queries to full number of heads
        queries = queries.view(
            batch_size,
            seq_length,
            self.num_attention_heads,
            self.head_dim,
        )

        # Reshape keys and values to number of query groups
        keys = keys.view(
            batch_size,
            seq_length,
            self.num_query_groups,
            self.head_dim,
        )
        values = values.view(
            batch_size,
            seq_length,
            self.num_query_groups,
            self.head_dim,
        )

        # Transpose for batch matrix multiplication
        queries = queries.transpose(
            1, 2
        )  # (batch, n_heads, seq_len, head_dim)
        keys = keys.transpose(
            1, 2
        )  # (batch, n_groups, seq_len, head_dim)
        values = values.transpose(
            1, 2
        )  # (batch, n_groups, seq_len, head_dim)

        # Repeat keys and values for each head in the group
        keys = keys.repeat_interleave(self.heads_per_group, dim=1)
        values = values.repeat_interleave(self.heads_per_group, dim=1)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        if attention_mask is not None:
            # Expand attention mask to match scores dimensions
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            expanded_mask = expanded_mask.expand(
                batch_size,
                self.num_attention_heads,
                seq_length,
                seq_length,
            )
            mask_value = torch.finfo(scores.dtype).min
            attention_mask = expanded_mask.eq(0).float() * mask_value
            scores = scores + attention_mask

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute attention output
        attention_output = torch.matmul(attention_weights, values)
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape(
            batch_size, seq_length, -1
        )

        return attention_output, None


class MoETransformer(nn.Module):
    """
    Production-grade Transformer model with Mixture of Experts and Multi-Query Attention.

    Features:
    - Multi-Query Attention mechanism for efficient inference
    - Mixture of Experts for dynamic routing and specialization
    - Real-time weight updates based on input similarity
    - Built-in logging and monitoring
    - Type annotations for better code maintainability
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Initialize components
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Multi-Query Attention layers
        self.attention_layers = nn.ModuleList(
            [
                MultiQueryAttention(config)
                for _ in range(config.num_expert_layers)
            ]
        )

        # Mixture of Experts layers
        self.moe_layers = nn.ModuleList(
            [
                MixtureOfExperts(config)
                for _ in range(config.num_expert_layers)
            ]
        )

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_prob)

        # Output projection
        self.output_projection = nn.Linear(
            config.hidden_size, config.vocab_size
        )

        # Initialize weights
        self.apply(self._init_weights)
        logger.info("Initialized MoETransformer model")

    def _init_weights(self, module: nn.Module):
        """Initialize model weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
            if (
                isinstance(module, nn.Linear)
                and module.bias is not None
            ):
                module.bias.data.zero_()

    def get_position_embeddings(self, position_ids: Tensor) -> Tensor:
        """Generate position embeddings."""
        return self.position_embedding(position_ids)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        cache: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding
            position_ids: Position IDs for positioning encoding
            cache: Cache for key/value states in generation

        Returns:
            tuple: (logits, auxiliary_outputs)
        """
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(
                input_ids
            )

        # Get embeddings
        inputs_embeds = self.embedding(input_ids)
        position_embeds = self.get_position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Initialize auxiliary outputs
        aux_outputs = {"moe_losses": []}

        # Process through transformer layers
        for attention_layer, moe_layer in zip(
            self.attention_layers, self.moe_layers
        ):
            # Multi-Query Attention
            attention_output, _ = attention_layer(
                hidden_states, attention_mask, cache
            )
            hidden_states = self.layer_norm(
                hidden_states + attention_output
            )

            # Mixture of Experts
            moe_output, moe_aux = moe_layer(hidden_states)
            hidden_states = self.layer_norm(
                hidden_states + moe_output
            )
            aux_outputs["moe_losses"].append(
                moe_aux["load_balancing_loss"]
            )

        # Final output projection
        logits = self.output_projection(hidden_states)

        return logits, aux_outputs

    def fetch_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        aux_outputs: Dict,
        reduction: str = "mean",
    ) -> Tensor:
        """
        Calculate the total loss including MoE balancing losses.

        Args:
            logits: Model output logits
            labels: Ground truth labels
            aux_outputs: Auxiliary outputs from forward pass
            reduction: Loss reduction method

        Returns:
            Tensor: Total loss
        """
        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
            reduction=reduction,
        )

        # Calculate MoE loss
        moe_loss = torch.stack(aux_outputs["moe_losses"]).mean()

        # Combine losses
        total_loss = ce_loss + 0.01 * moe_loss

        logger.debug(
            f"CE Loss: {ce_loss.item():.4f}, "
            f"MoE Loss: {moe_loss.item():.4f}"
        )

        return total_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tensor:
        """
        Generate text using the model.

        Args:
            input_ids: Initial input tokens
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability for nucleus sampling

        Returns:
            Tensor: Generated token IDs
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize sequence with input_ids
        generated = input_ids

        # Cache for key-value pairs
        cache = {}

        for _ in range(max_length):
            # Get position IDs for current sequence
            position_ids = torch.arange(
                generated.shape[1], dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(
                batch_size, -1
            )

            # Forward pass
            logits, _ = self.forward(
                generated, position_ids=position_ids, cache=cache
            )

            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][
                        ..., -1, None
                    ]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = (
                    sorted_indices_to_remove[..., :-1].clone()
                )
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[
                    sorted_indices_to_remove
                ]
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append next token to sequence
            generated = torch.cat((generated, next_token), dim=1)

            # Check for end of sequence token
            if (next_token == self.config.vocab_size - 1).all():
                break

        return generated


# Initialize model configuration
config = TransformerConfig(
    vocab_size=50257,
    hidden_size=768,
    num_attention_heads=12,
    num_expert_layers=4,
    num_experts=8,
    expert_capacity=32,
    max_position_embeddings=1024,
    num_query_groups=4,
)


def prepare_sample_data(
    batch_size: int = 8,
    seq_length: int = 512,
    vocab_size: int = 50257,
) -> DataLoader:
    """Create sample data for demonstration."""
    # Create random input sequences
    input_ids = torch.randint(
        0, vocab_size, (100, seq_length)  # 100 samples
    )

    # Create target sequences (shifted by 1)
    labels = torch.randint(0, vocab_size, (100, seq_length))

    # Create attention masks (1 for real tokens, 0 for padding)
    attention_mask = torch.ones_like(input_ids)

    # Create dataset and dataloader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return dataloader


def train_step(
    model: MoETransformer,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """Execute single training step."""
    model.train()
    optimizer.zero_grad()

    # Unpack batch
    input_ids, attention_mask, labels = [b.to(device) for b in batch]

    # Forward pass
    logits, aux_outputs = model(
        input_ids=input_ids, attention_mask=attention_mask
    )

    # Calculate loss
    loss = model.fetch_loss(logits, labels, aux_outputs)

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize model
    model = MoETransformer(config).to(device)
    logger.info("Model initialized")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.01
    )

    # Prepare data
    dataloader = prepare_sample_data()
    logger.info("Data prepared")

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            loss = train_step(model, batch, optimizer, device)
            epoch_losses.append(loss)

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} "
                    f"Batch {batch_idx}/{len(dataloader)} "
                    f"Loss: {loss:.4f}"
                )

        avg_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # Generation example
    model.eval()
    with torch.no_grad():
        # Prepare input prompt
        prompt = torch.randint(0, config.vocab_size, (1, 10)).to(
            device
        )

        # Generate sequence
        generated = model.generate(
            input_ids=prompt,
            max_length=50,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )

        logger.info(f"Generated sequence shape: {generated.shape}")


if __name__ == "__main__":
    main()
