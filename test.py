import torch
import torch.nn as nn
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from loguru import logger
import math


@dataclass
class StarAttentionConfig:
    """Configuration for StarAttention module.

    Attributes:
        hidden_size: Dimension of the model's hidden states
        num_attention_heads: Number of attention heads
        num_hosts: Number of hosts in the distributed system
        block_size: Size of each context block
        anchor_size: Size of the anchor block
        dropout_prob: Dropout probability (default: 0.1)
        layer_norm_eps: Layer normalization epsilon (default: 1e-12)
    """

    hidden_size: int
    num_attention_heads: int
    num_hosts: int
    block_size: int
    anchor_size: int
    dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12


class StarAttention(nn.Module):
    """
    Implementation of Star Attention mechanism for distributed inference.

    The module implements a two-phase attention mechanism:
    1. Local Context Encoding with Anchor Blocks
    2. Query Encoding and Output Generation with Global Attention
    """

    def __init__(self, config: StarAttentionConfig):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} not divisible by number of attention "
                f"heads {config.num_attention_heads}"
            )

        self.config = config
        self.head_dim = (
            config.hidden_size // config.num_attention_heads
        )

        # Initialize components
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        # KV cache for storing computed key/value pairs
        self.kv_cache = {}

        logger.info(
            f"Initialized StarAttention with config: {config}"
        )

    def _split_heads(
        self, tensor: torch.Tensor, num_heads: int
    ) -> torch.Tensor:
        """Split the last dimension into (num_heads, head_dim)."""
        batch_size, seq_len, _ = tensor.size()
        tensor = tensor.view(
            batch_size, seq_len, num_heads, self.head_dim
        )
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Merge the head dimension back into hidden_size."""
        batch_size, _, seq_len, _ = tensor.size()
        tensor = tensor.transpose(1, 2)
        return tensor.reshape(
            batch_size, seq_len, self.config.hidden_size
        )

    def _compute_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention scores and weighted values."""
        # Scale dot-product attention
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Online softmax computation
        attention_probs = torch.nn.functional.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)

        return context, attention_probs

    def phase1_local_context_encoding(
        self,
        input_ids: torch.Tensor,
        host_id: int,
        device: Union[str, torch.device] = "cuda",
    ) -> None:
        """
        Phase 1: Local Context Encoding with Anchor Blocks

        Args:
            input_ids: Input tensor of shape (batch_size, seq_len)
            host_id: ID of the current host
            device: Device to run computations on
        """
        logger.debug(f"Starting Phase 1 on host {host_id}")

        # Calculate block assignments
        block_start = host_id * self.config.block_size
        block_end = block_start + self.config.block_size

        # Get local block
        local_block = input_ids[:, block_start:block_end].to(device)

        # Get anchor block (first block)
        anchor_block = input_ids[:, : self.config.anchor_size].to(
            device
        )

        # Compute KV pairs for local block
        local_hidden = self.layer_norm(local_block)
        local_key = self._split_heads(
            self.key(local_hidden), self.config.num_attention_heads
        )
        local_value = self._split_heads(
            self.value(local_hidden), self.config.num_attention_heads
        )

        # Store in KV cache
        self.kv_cache[host_id] = {
            "key": local_key,
            "value": local_value,
            "anchor_key": (
                None
                if host_id == 0
                else self._split_heads(
                    self.key(self.layer_norm(anchor_block)),
                    self.config.num_attention_heads,
                )
            ),
        }

        logger.debug(
            f"Phase 1 complete on host {host_id}. KV cache shapes - "
            f"key: {local_key.shape}, value: {local_value.shape}"
        )

    def phase2_query_encoding(
        self,
        query_input: torch.Tensor,
        host_id: int,
        is_query_host: bool,
        device: Union[str, torch.device] = "cuda",
    ) -> Optional[torch.Tensor]:
        """
        Phase 2: Query Encoding and Output Generation

        Args:
            query_input: Query tensor of shape (batch_size, seq_len, hidden_size)
            host_id: ID of the current host
            is_query_host: Whether this host is the query host
            device: Device to run computations on

        Returns:
            Output tensor if this is the query host, None otherwise
        """
        logger.debug(f"Starting Phase 2 on host {host_id}")

        # Transform query
        query_hidden = self.layer_norm(query_input)
        query = self._split_heads(
            self.query(query_hidden), self.config.num_attention_heads
        )

        # Compute local attention scores
        local_context, local_probs = self._compute_attention_scores(
            query,
            self.kv_cache[host_id]["key"],
            self.kv_cache[host_id]["value"],
        )

        if not is_query_host:
            # Non-query hosts send their local attention statistics
            dist.send(local_probs, dst=self.config.num_hosts - 1)
            return None

        # Query host aggregates attention from all hosts
        all_attention_probs = [local_probs]
        for src_rank in range(self.config.num_hosts - 1):
            probs = torch.empty_like(local_probs)
            dist.recv(probs, src=src_rank)
            all_attention_probs.append(probs)

        # Compute global attention
        torch.mean(torch.stack(all_attention_probs), dim=0)

        # Final output computation
        output = self._merge_heads(local_context)
        output = self.dropout(output)

        logger.debug(
            f"Phase 2 complete on host {host_id}. Output shape: {output.shape}"
        )

        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        query_input: torch.Tensor,
        host_id: int,
        is_query_host: bool,
        device: Union[str, torch.device] = "cuda",
    ) -> Optional[torch.Tensor]:
        """
        Forward pass of the StarAttention module.

        Args:
            input_ids: Input tensor of shape (batch_size, seq_len)
            query_input: Query tensor of shape (batch_size, seq_len, hidden_size)
            host_id: ID of the current host
            is_query_host: Whether this host is the query host
            device: Device to run computations on

        Returns:
            Output tensor if this is the query host, None otherwise
        """
        # Phase 1: Local Context Encoding
        self.phase1_local_context_encoding(input_ids, host_id, device)

        # Phase 2: Query Encoding and Output Generation
        return self.phase2_query_encoding(
            query_input, host_id, is_query_host, device
        )


# Example forward pass
config = StarAttentionConfig(
    hidden_size=768,
    num_attention_heads=12,
    num_hosts=3,
    block_size=512,
    anchor_size=128,
)

# Initialize model
model = StarAttention(config)

# Example input tensors
batch_size = 4
seq_len = 512
input_ids = torch.randint(
    0, 1000, (batch_size, seq_len)
)  # Random input IDs
query_input = torch.randn(
    batch_size, seq_len, config.hidden_size
)  # Random query input

# Example forward pass for query host (host_id = 2)
output = model(
    input_ids=input_ids,
    query_input=query_input,
    host_id=2,
    is_query_host=True,
    device="cpu",
)

print(output)
