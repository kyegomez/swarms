from enum import Enum
from typing import Union, Optional
import io
from PIL import Image
import numpy as np
import torch
import struct
import magic


from enum import auto
from typing import List, Dict, Tuple
import wave
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from einops import rearrange
from torch import Tensor


@dataclass
class ModelConfig:
    """Configuration for the enhanced BytePredictor model."""

    vocab_size: int = 256  # Standard byte range
    hidden_size: int = 1024
    num_layers: int = 12
    num_key_value_heads: int = 8  # For multi-query attention
    num_query_heads: int = 32  # More query heads than kv heads
    dropout: float = 0.1
    max_sequence_length: int = 8192
    rope_theta: float = 10000.0
    layer_norm_eps: float = 1e-5
    vocab_parallel: bool = False
    qk_norm: bool = True
    qk_norm_scale: float = None
    attention_bias: bool = False


class MultiQueryAttention(nn.Module):
    """Fixed Multi-Query Attention implementation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_query_heads = config.num_query_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_query_heads
        self.qk_scale = config.qk_norm_scale or (self.head_dim**-0.5)

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_query_heads * self.head_dim
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
        )
        self.o_proj = nn.Linear(
            config.num_query_heads * self.head_dim, config.hidden_size
        )

        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        # Project and reshape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [seq_len, batch, heads, head_dim]
        q = q.view(
            batch_size,
            seq_length,
            self.num_query_heads,
            self.head_dim,
        ).permute(1, 0, 2, 3)
        k = k.view(
            batch_size,
            seq_length,
            self.num_key_value_heads,
            self.head_dim,
        ).permute(1, 0, 2, 3)
        v = v.view(
            batch_size,
            seq_length,
            self.num_key_value_heads,
            self.head_dim,
        ).permute(1, 0, 2, 3)

        # Apply rotary embeddings
        # q, k = self.rotary(q, k, seq_length)

        # Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Handle MQA head expansion
        if self.num_key_value_heads != self.num_query_heads:
            k = k.repeat_interleave(
                self.num_query_heads // self.num_key_value_heads,
                dim=2,
            )
            v = v.repeat_interleave(
                self.num_query_heads // self.num_key_value_heads,
                dim=2,
            )

        # Compute attention
        # Reshape for matmul: [batch, heads, seq_length, head_dim]
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        attn_weights = (
            torch.matmul(q, k.transpose(-2, -1)) * self.qk_scale
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, v)

        # Reshape back to [batch, seq_length, hidden_size]
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, -1)
        )
        output = self.o_proj(output)

        return output


class EnhancedBytePredictor(nn.Module):
    """Enhanced byte prediction model with state-of-the-art features."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": MultiQueryAttention(config),
                        "attention_norm": nn.LayerNorm(
                            config.hidden_size,
                            eps=config.layer_norm_eps,
                        ),
                        "feed_forward": nn.Sequential(
                            nn.Linear(
                                config.hidden_size,
                                4 * config.hidden_size,
                            ),
                            nn.GELU(),
                            nn.Linear(
                                4 * config.hidden_size,
                                config.hidden_size,
                            ),
                        ),
                        "feed_forward_norm": nn.LayerNorm(
                            config.hidden_size,
                            eps=config.layer_norm_eps,
                        ),
                    }
                )
                for _ in range(config.num_layers)
            ]
        )

        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.output = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)
            attention_mask: Optional attention mask

        Returns:
            Tensor of logits with shape (batch_size, sequence_length, vocab_size)
        """
        hidden_states = self.tok_embeddings(input_ids)

        # Create causal mask if needed
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.ones(
                    (input_ids.size(1), input_ids.size(1)),
                    device=input_ids.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            attention_mask = attention_mask.masked_fill(
                attention_mask == 1, float("-inf")
            )

        # Apply transformer layers
        for layer in self.layers:
            # Attention block
            hidden_states = hidden_states + layer["attention"](
                layer["attention_norm"](hidden_states), attention_mask
            )

            # Feed-forward block
            hidden_states = hidden_states + layer["feed_forward"](
                layer["feed_forward_norm"](hidden_states)
            )

        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        return logits

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross entropy loss.

        Args:
            input_ids: Input token ids
            target_ids: Target token ids
            attention_mask: Optional attention mask

        Returns:
            Loss value
        """
        logits = self(input_ids, attention_mask)
        loss = F.cross_entropy(
            rearrange(logits, "b s v -> (b s) v"),
            rearrange(target_ids, "b s -> (b s)"),
        )
        return loss

    @torch.no_grad()
    def _generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            input_ids: Starting sequence
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: K for top-k sampling
            top_p: P for nucleus sampling
            repetition_penalty: Penalty for repeating tokens

        Returns:
            Generated sequence
        """
        batch_size, seq_length = input_ids.shape
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            if generated.size(1) >= self.config.max_sequence_length:
                break

            # Forward pass
            logits = self(generated)[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        logits[i, token_id] /= repetition_penalty

            # Apply top-k sampling
            if top_k is not None:
                indices_to_remove = (
                    logits
                    < torch.topk(logits, top_k)[0][..., -1, None]
                )
                logits[indices_to_remove] = float("-inf")

            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True
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

                indices_to_remove = torch.zeros_like(
                    logits, dtype=torch.bool
                )
                indices_to_remove.scatter_(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
    ):
        tensor_data = self._generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        return tensor_to_data(tensor_data)


# import torch
# from typing import Optional


class DataType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    BINARY = "binary"


class ByteDetokenizer:
    """Utility class for converting model output bytes back to original data formats."""

    @staticmethod
    def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
        """Convert model output tensor to bytes."""
        # Convert logits/probabilities to byte values
        if tensor.dim() > 1:
            # If we have logits, convert to byte indices
            byte_indices = tensor.argmax(dim=-1)
        else:
            byte_indices = tensor

        # Convert to Python bytes
        return bytes(
            byte_indices.cpu().numpy().astype(np.uint8).tolist()
        )

    @staticmethod
    def decode_text(byte_sequence: bytes) -> str:
        """Convert bytes to text."""
        try:
            return byte_sequence.decode("utf-8")
        except UnicodeDecodeError:
            # Try with error handling
            return byte_sequence.decode("utf-8", errors="replace")

    @staticmethod
    def decode_image(
        byte_sequence: bytes,
        mode: str = "RGB",
        size: Optional[tuple] = None,
    ) -> Image.Image:
        """Convert bytes to image.

        Args:
            byte_sequence: Raw image bytes
            mode: Image mode (RGB, RGBA, L, etc.)
            size: Optional tuple of (width, height)
        """
        try:
            # Try to load as-is first (for standard image formats)
            img = Image.open(io.BytesIO(byte_sequence))
            if size:
                img = img.resize(size)
            return img
        except:
            # If failed, assume raw pixel data
            if not size:
                # Try to determine size from byte count
                pixel_count = len(byte_sequence) // len(mode)
                size = (
                    int(np.sqrt(pixel_count)),
                    int(np.sqrt(pixel_count)),
                )

            # Convert raw bytes to pixel array
            pixels = np.frombuffer(byte_sequence, dtype=np.uint8)
            pixels = pixels.reshape((*size, len(mode)))

            return Image.fromarray(pixels, mode=mode)

    @staticmethod
    def decode_audio(
        byte_sequence: bytes,
        sample_rate: int = 44100,
        channels: int = 2,
        sample_width: int = 2,
    ) -> np.ndarray:
        """Convert bytes to audio samples.

        Args:
            byte_sequence: Raw audio bytes
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            sample_width: Bytes per sample (1, 2, or 4)
        """
        # Determine format string based on sample width
        format_str = {
            1: "b",  # signed char
            2: "h",  # short
            4: "i",  # int
        }[sample_width]

        # Unpack bytes to samples
        sample_count = len(byte_sequence) // (channels * sample_width)
        samples = struct.unpack(
            f"<{sample_count * channels}{format_str}", byte_sequence
        )

        # Reshape to [samples, channels]
        return np.array(samples).reshape(-1, channels)

    def decode_data(
        self,
        model_output: Union[torch.Tensor, bytes],
        data_type: DataType,
        **kwargs,
    ) -> Union[str, Image.Image, np.ndarray, bytes]:
        """Main method to decode model output to desired format.

        Args:
            model_output: Either tensor from model or raw bytes
            data_type: Type of data to decode to
            **kwargs: Additional parameters for specific decoders

        Returns:
            Decoded data in specified format
        """
        # Convert tensor to bytes if needed
        if isinstance(model_output, torch.Tensor):
            byte_sequence = self.tensor_to_bytes(model_output)
        else:
            byte_sequence = model_output

        # Decode based on type
        if data_type == DataType.TEXT:
            return self.decode_text(byte_sequence)
        elif data_type == DataType.IMAGE:
            return self.decode_image(byte_sequence, **kwargs)
        elif data_type == DataType.AUDIO:
            return self.decode_audio(byte_sequence, **kwargs)
        elif data_type == DataType.VIDEO:
            raise NotImplementedError(
                "Video decoding not yet implemented"
            )
        else:  # BINARY
            return byte_sequence


# Usage example


class Modality(Enum):
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    BINARY = auto()
    MULTIMODAL = auto()


@dataclass
class ModalityInfo:
    """Information about detected modality."""

    modality: Modality
    confidence: float
    metadata: Dict[str, any]
    sub_modalities: Optional[List["ModalityInfo"]] = None


class ModalityDetector:
    """Detects data modalities from byte sequences."""

    # Common file signatures (magic numbers)
    SIGNATURES = {
        # Images
        b"\xFF\xD8\xFF": "JPEG",
        b"\x89PNG\r\n\x1a\n": "PNG",
        b"GIF87a": "GIF",
        b"GIF89a": "GIF",
        b"RIFF": "WEBP",
        # Audio
        b"RIFF....WAVE": "WAV",
        b"ID3": "MP3",
        b"\xFF\xFB": "MP3",
        b"OggS": "OGG",
        # Video
        b"\x00\x00\x00\x18ftypmp42": "MP4",
        b"\x00\x00\x00\x1Cftypav01": "MP4",
        b"\x1A\x45\xDF\xA3": "WEBM",
    }

    def __init__(self):
        self.magic = magic.Magic(mime=True)

    def _check_text_probability(self, data: bytes) -> float:
        """Estimate probability that data is text."""
        # Check if data is valid UTF-8
        try:
            data.decode("utf-8")
            # Count printable ASCII characters
            printable = sum(1 for b in data if 32 <= b <= 126)
            return printable / len(data)
        except UnicodeDecodeError:
            return 0.0

    def _check_image_validity(self, data: bytes) -> Tuple[bool, Dict]:
        """Check if data is a valid image and extract metadata."""
        try:
            with io.BytesIO(data) as bio:
                img = Image.open(bio)
                return True, {
                    "format": img.format,
                    "size": img.size,
                    "mode": img.mode,
                }
        except:
            return False, {}

    def _check_audio_validity(self, data: bytes) -> Tuple[bool, Dict]:
        """Check if data is valid audio and extract metadata."""
        try:
            with io.BytesIO(data) as bio:
                # Try to parse as WAV
                with wave.open(bio) as wav:
                    return True, {
                        "channels": wav.getnchannels(),
                        "sample_width": wav.getsampwidth(),
                        "framerate": wav.getframerate(),
                        "frames": wav.getnframes(),
                    }
        except:
            # Check for other audio signatures
            for sig in [b"ID3", b"\xFF\xFB", b"OggS"]:
                if data.startswith(sig):
                    return True, {"format": "compressed_audio"}
            return False, {}

    def _detect_boundaries(
        self, data: bytes
    ) -> List[Tuple[int, int, Modality]]:
        """Detect boundaries between different modalities in the data."""
        boundaries = []
        current_pos = 0

        while current_pos < len(data):
            # Look for known signatures
            for sig, format_type in self.SIGNATURES.items():
                if data[current_pos:].startswith(sig):
                    # Found a signature, determine its length
                    if format_type in ["JPEG", "PNG", "GIF"]:
                        # Find image end
                        try:
                            with io.BytesIO(
                                data[current_pos:]
                            ) as bio:
                                img = Image.open(bio)
                                img.verify()
                                size = bio.tell()
                                boundaries.append(
                                    (
                                        current_pos,
                                        current_pos + size,
                                        Modality.IMAGE,
                                    )
                                )
                                current_pos += size
                                continue
                        except:
                            pass

            # Check for text sections
            text_prob = self._check_text_probability(
                data[current_pos : current_pos + 1024]
            )
            if text_prob > 0.8:
                # Look for end of text section
                end_pos = current_pos + 1
                while end_pos < len(data):
                    if (
                        self._check_text_probability(
                            data[end_pos : end_pos + 32]
                        )
                        < 0.5
                    ):
                        break
                    end_pos += 1
                boundaries.append(
                    (current_pos, end_pos, Modality.TEXT)
                )
                current_pos = end_pos
                continue

            current_pos += 1

        return boundaries

    def detect_modality(self, data: bytes) -> ModalityInfo:
        """Detect modality of byte sequence."""
        # First check for single modality
        mime_type = self.magic.from_buffer(data)

        # Check text
        text_prob = self._check_text_probability(data)
        if text_prob > 0.9:
            return ModalityInfo(
                modality=Modality.TEXT,
                confidence=text_prob,
                metadata={"mime_type": mime_type},
            )

        # Check image
        is_image, image_meta = self._check_image_validity(data)
        if is_image:
            return ModalityInfo(
                modality=Modality.IMAGE,
                confidence=1.0,
                metadata={**image_meta, "mime_type": mime_type},
            )

        # Check audio
        is_audio, audio_meta = self._check_audio_validity(data)
        if is_audio:
            return ModalityInfo(
                modality=Modality.AUDIO,
                confidence=1.0,
                metadata={**audio_meta, "mime_type": mime_type},
            )

        # Check for multimodal content
        boundaries = self._detect_boundaries(data)
        if len(boundaries) > 1:
            sub_modalities = []
            for start, end, modality in boundaries:
                chunk_data = data[start:end]
                sub_info = self.detect_modality(chunk_data)
                if sub_info.modality != Modality.BINARY:
                    sub_modalities.append(sub_info)

            if sub_modalities:
                return ModalityInfo(
                    modality=Modality.MULTIMODAL,
                    confidence=0.8,
                    metadata={"mime_type": "multipart/mixed"},
                    sub_modalities=sub_modalities,
                )

        # Default to binary
        return ModalityInfo(
            modality=Modality.BINARY,
            confidence=0.5,
            metadata={"mime_type": mime_type},
        )

    def split_modalities(
        self, data: bytes
    ) -> List[Tuple[Modality, bytes, Dict]]:
        """Split multimodal data into separate modalities."""
        boundaries = self._detect_boundaries(data)
        result = []

        for start, end, modality in boundaries:
            chunk = data[start:end]
            info = self.detect_modality(chunk)
            result.append((modality, chunk, info.metadata))

        return result


class AutoDetectBytesDecoder:
    """Decoder that automatically detects and decodes different modalities."""

    def __init__(self):
        self.detector = ModalityDetector()
        self.text_decoder = ByteDetokenizer()  # From previous example

    def decode(
        self, data: bytes
    ) -> Union[str, Image.Image, np.ndarray, List[any]]:
        """Automatically detect and decode byte sequence."""
        info = self.detector.detect_modality(data)

        if info.modality == Modality.MULTIMODAL:
            # Handle multimodal content
            parts = self.detector.split_modalities(data)
            return [
                self.decode(chunk) for modality, chunk, _ in parts
            ]

        if info.modality == Modality.TEXT:
            return self.text_decoder.decode_text(data)
        elif info.modality == Modality.IMAGE:
            return self.text_decoder.decode_image(data)
        elif info.modality == Modality.AUDIO:
            return self.text_decoder.decode_audio(data)
        else:
            return data


# # Example usage
# def demo_auto_detection():
#     """Demonstrate auto modality detection."""
#     # Create mixed content
#     text = "Hello, World!".encode('utf-8')

#     # Create a small test image
#     img = Image.new('RGB', (100, 100), color='red')
#     img_bytes = io.BytesIO()
#     img.save(img_bytes, format='PNG')

#     # Combine into multimodal content
#     mixed_content = text + img_bytes.getvalue()

#     # Initialize decoder
#     decoder = AutoDetectBytesDecoder()

#     # Decode
#     result = decoder.decode(mixed_content)

#     if isinstance(result, list):
#         print("Detected multimodal content:")
#         for i, part in enumerate(result):
#             print(f"Part {i+1}: {type(part)}")

# if __name__ == "__main__":
#     demo_auto_detection()


def tensor_to_data(tensor: Tensor):
    byte_sequence = ByteDetokenizer.tensor_to_bytes(tensor)

    # Initialize auto-detector
    decoder = AutoDetectBytesDecoder()

    # Decode with automatic detection
    result = decoder.decode(byte_sequence)

    return result


def demo_byte_predictor():
    """Demo with smaller dimensions to test."""
    # Initialize model configuration with adjusted dimensions
    config = ModelConfig(
        vocab_size=256,
        hidden_size=128,  # Smaller for testing
        num_layers=2,  # Fewer layers for testing
        num_key_value_heads=2,
        num_query_heads=4,
        dropout=0.1,
        max_sequence_length=1024,
    )

    # Initialize model
    model = EnhancedBytePredictor(config)
    logger.info("Model initialized")

    # Move to GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    logger.info(f"Using device: {device}")

    # Create sample input data
    batch_size = 2
    seq_length = 16  # Shorter sequence for testing
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_length), device=device
    )
    logger.info(f"Created input tensor of shape: {input_ids.shape}")

    # Test forward pass
    try:
        logits = model(input_ids)
        logger.info(
            f"Forward pass successful! Output shape: {logits.shape}"
        )

        # Test loss computation
        target_ids = torch.randint(
            0,
            config.vocab_size,
            (batch_size, seq_length),
            device=device,
        )
        loss = model.compute_loss(input_ids, target_ids)
        logger.info(
            f"Loss computation successful! Loss value: {loss.item():.4f}"
        )

        # Test generation
        prompt = torch.randint(
            0,
            config.vocab_size,
            (1, 4),  # Very short prompt for testing
            device=device,
        )
        generated = model.generate(
            prompt, max_new_tokens=8, temperature=0.8, top_k=50
        )
        logger.info(
            f"Generation successful! Generated shape: {generated.shape}"
        )

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    # Set up logging
    # logger.remove()  # Remove default handler
    # logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | {level} | {message}")

    demo_byte_predictor()
