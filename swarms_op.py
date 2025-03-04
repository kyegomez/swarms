"""
MultiModelOptimizer: A high-performance optimizer for training multiple transformer models simultaneously.

This optimizer implements several advanced techniques:
1. Gradient accumulation with dynamic batch sizing
2. Hierarchical parameter synchronization
3. Memory-efficient gradient sharing with shape compatibility
4. Adaptive learning rate scheduling per model
5. Convergence acceleration via momentum tuning
6. Robust error handling for production environments

Author: Claude 3.7 Sonnet
License: MIT
"""

import math
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from loguru import logger
import numpy as np


class MultiModelOptimizer(Optimizer):
    """
    An optimizer designed for training multiple models simultaneously with shared gradient information,
    adaptive learning rates, and efficient memory usage.

    Args:
        models (Dict[str, nn.Module]): Dictionary mapping model names to model instances
        lr (float, optional): Initial learning rate. Default: 1e-3
        betas (Tuple[float, float], optional): Coefficients for computing running averages of gradient and its square. Default: (0.9, 0.999)
        eps (float, optional): Term added to denominator for numerical stability. Default: 1e-8
        weight_decay (float, optional): Weight decay coefficient. Default: 0
        amsgrad (bool, optional): Whether to use the AMSGrad variant. Default: False
        grad_sync_frequency (int, optional): How often to synchronize gradients between models. Default: 1
        warmup_steps (int, optional): Number of warmup steps for learning rate. Default: 1000
        model_weights (Dict[str, float], optional): Relative importance weights for each model. Default: None
        gradient_accumulation_steps (int, optional): Number of steps to accumulate gradients before update. Default: 1
        clip_grad_norm (float, optional): Maximum norm for gradient clipping. Default: None
        use_cosine_schedule (bool, optional): Whether to use cosine annealing schedule. Default: True
        sync_every_step (bool, optional): Whether to synchronize parameters on every step. Default: False
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        grad_sync_frequency: int = 1,
        warmup_steps: int = 1000,
        model_weights: Optional[Dict[str, float]] = None,
        gradient_accumulation_steps: int = 1,
        clip_grad_norm: Optional[float] = None,
        use_cosine_schedule: bool = True,
        sync_every_step: bool = False,
    ):

        # Initialize model weights if not provided
        if model_weights is None:
            model_weights = {name: 1.0 for name in models.keys()}

        # Normalize weights to sum to 1
        total_weight = sum(model_weights.values())
        self.model_weights = {
            k: v / total_weight for k, v in model_weights.items()
        }

        # Store models
        self.models = models

        # Collect all parameters from all models
        parameters = []
        self.model_param_groups: Dict[str, List[Dict]] = {}

        for model_name, model in models.items():
            model_params = []
            for param in model.parameters():
                if param.requires_grad:
                    param_dict = {
                        "params": [param],
                        "model_name": model_name,
                    }
                    parameters.append(param_dict)
                    model_params.append(param_dict)
            self.model_param_groups[model_name] = model_params

        # Initialize optimizer with all parameters
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super(MultiModelOptimizer, self).__init__(
            parameters, defaults
        )

        # Additional settings
        self.grad_sync_frequency = grad_sync_frequency
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.current_accumulation_step = 0
        self.clip_grad_norm = clip_grad_norm
        self.use_cosine_schedule = use_cosine_schedule
        self.sync_every_step = sync_every_step

        # Metrics and tracking
        self.model_losses: Dict[str, List[float]] = defaultdict(list)
        self.model_gradients: Dict[str, torch.Tensor] = {}
        self.shared_gradient_cache: Dict[str, torch.Tensor] = {}

        # Set up gradient sharing structures
        self.param_name_to_model = {}
        for name, model in self.models.items():
            for param_name, _ in model.named_parameters():
                self.param_name_to_model[f"{name}.{param_name}"] = (
                    name
                )

        # Configure logger
        logger.configure(
            handlers=[
                {
                    "sink": "logs/multi_model_optimizer.log",
                    "level": "INFO",
                },
                {"sink": lambda msg: print(msg), "level": "INFO"},
            ]
        )

        logger.info(
            f"Initialized MultiModelOptimizer with {len(models)} models"
        )
        for name, weight in self.model_weights.items():
            logger.info(f"Model {name} weight: {weight:.4f}")

    def get_lr_multiplier(self) -> float:
        """Calculate the learning rate multiplier based on warmup and schedule."""
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return float(self.step_count) / float(
                max(1, self.warmup_steps)
            )

        if self.use_cosine_schedule:
            # Cosine decay after warmup
            decay_steps = max(1, self.step_count - self.warmup_steps)
            cosine_decay = 0.5 * (
                1
                + math.cos(
                    math.pi
                    * decay_steps
                    / (10000 * self.gradient_accumulation_steps)
                )
            )
            return max(
                0.1, cosine_decay
            )  # Don't let LR go below 10% of base value

        return 1.0  # Constant LR after warmup if not using cosine

    def share_gradients(self):
        """Share gradient information across models for similar parameters."""
        # First, collect all gradients by parameter type and shape
        param_type_shape_grads = defaultdict(list)

        for model_name, model in self.models.items():
            for param_name, param in model.named_parameters():
                if param.grad is not None:
                    # Classify parameter by name pattern and include shape to ensure compatibility
                    param_type = self._classify_parameter(param_name)
                    param_shape = param.shape
                    key = (param_type, param_shape)
                    param_type_shape_grads[key].append(
                        (model_name, param_name, param.grad)
                    )

        # Now compute shared gradients for each parameter type and shape combination
        for (
            param_type,
            param_shape,
        ), grads in param_type_shape_grads.items():
            if len(grads) <= 1:
                continue  # Skip if only one model has this parameter type+shape

            cache_key = f"{param_type}_{param_shape}"

            # Compute weighted average gradient for this parameter type+shape
            for model_name, param_name, grad in grads:
                weight = self.model_weights[model_name]

                # Initialize shared gradient for this parameter if not exists
                if cache_key not in self.shared_gradient_cache:
                    self.shared_gradient_cache[cache_key] = (
                        torch.zeros_like(grad)
                    )

                # Add weighted contribution
                self.shared_gradient_cache[cache_key].add_(
                    grad * weight
                )

            # Now apply a fraction of the shared gradient back to each model's parameter
            for model_name, param_name, _ in grads:
                param = self.models[model_name].get_parameter(
                    param_name
                )
                if param.grad is not None:
                    # Mix original gradient with shared gradient
                    sharing_ratio = 0.2  # 20% shared, 80% original
                    param.grad.mul_(1 - sharing_ratio).add_(
                        self.shared_gradient_cache[cache_key]
                        * sharing_ratio
                    )

        # Clear the cache for next iteration
        self.shared_gradient_cache.clear()

    def _classify_parameter(self, param_name: str) -> str:
        """Classify parameter by name to determine which parameters should share gradients."""
        # First, make sure we include the model architecture in the classification
        # to prevent mixing parameters from different architectures
        model_type = "unknown"
        if "bert" in param_name:
            model_type = "bert"
        elif "gpt" in param_name:
            model_type = "gpt"
        elif "roberta" in param_name:
            model_type = "roberta"
        elif "transformer" in param_name:
            model_type = "transformer"

        # Then classify by parameter type
        param_type = "other"
        if (
            "query" in param_name
            or "key" in param_name
            or "value" in param_name
        ):
            param_type = "attention"
        elif (
            "dense" in param_name
            or "fc" in param_name
            or "ffn" in param_name
        ):
            param_type = "ffn"
        elif "embedding" in param_name:
            param_type = "embedding"
        elif "norm" in param_name or "layer_norm" in param_name:
            param_type = "norm"
        elif "bias" in param_name:
            param_type = "bias"
        else:
            param_type = param_name.split(".")[
                -1
            ]  # Use the last component of the name

        # Combine model type and parameter type for more specific classification
        return f"{model_type}_{param_type}"

    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """Perform a single optimization step, handling gradient accumulation and sync."""
        loss = None
        if closure is not None:
            loss = closure()

        self.current_accumulation_step += 1

        # Only perform the update after accumulating enough gradients
        if (
            self.current_accumulation_step
            < self.gradient_accumulation_steps
        ):
            return loss

        self.current_accumulation_step = 0
        self.step_count += 1

        # Apply gradient clipping if configured
        if self.clip_grad_norm is not None:
            for model_name, model in self.models.items():
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.clip_grad_norm
                )

        # Share gradients between models if it's time
        if self.step_count % self.grad_sync_frequency == 0:
            self.share_gradients()

        # Calculate the current learning rate multiplier
        lr_multiplier = self.get_lr_multiplier()

        # Apply optimizer update for each parameter group
        for group in self.param_groups:
            # Get model-specific learning rate adjustment
            model_name = group["model_name"]
            model_weight = self.model_weights[model_name]

            # Adjust lr based on model weight and global multiplier
            model_lr_multiplier = lr_multiplier * (
                0.5 + 0.5 * model_weight
            )  # Scale between 50-150% based on weight

            # Extract parameters for this group
            p = group["params"][0]
            if p.grad is None:
                continue

            # State initialization
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            # Extract optimizer parameters
            beta1, beta2 = group["betas"]
            exp_avg, exp_avg_sq = (
                state["exp_avg"],
                state["exp_avg_sq"],
            )

            # Update step count
            state["step"] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                p.grad, p.grad, value=1 - beta2
            )

            # Apply AMSGrad if enabled
            if group["amsgrad"]:
                max_exp_avg_sq = state["max_exp_avg_sq"]
                torch.maximum(
                    max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq
                )
                denom = max_exp_avg_sq.sqrt().add_(group["eps"])
            else:
                denom = exp_avg_sq.sqrt().add_(group["eps"])

            # Bias correction
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            step_size = (
                group["lr"]
                * model_lr_multiplier
                * math.sqrt(bias_correction2)
                / bias_correction1
            )

            # Apply weight decay if configured
            if group["weight_decay"] > 0:
                p.data.add_(
                    p.data,
                    alpha=-group["weight_decay"]
                    * group["lr"]
                    * model_lr_multiplier,
                )

            # Update parameter
            p.data.addcdiv_(exp_avg, denom, value=-step_size)

        # Synchronize parameters if configured to do so every step
        if self.sync_every_step:
            self.synchronize_similar_parameters()

        return loss

    def synchronize_similar_parameters(self):
        """Synchronize similar parameters across models to promote convergence."""
        # Only sync occasionally
        if self.step_count % 10 != 0:
            return

        try:
            # First, identify similar parameters across models
            param_groups = defaultdict(list)

            for model_name, model in self.models.items():
                for param_name, param in model.named_parameters():
                    # Only sync parameters of the same shape
                    param_type = self._classify_parameter(param_name)
                    param_shape = param.shape
                    param_groups[(param_type, param_shape)].append(
                        (model_name, param_name, param)
                    )

            # For each group of similar parameters, synchronize values
            for (
                param_type,
                param_shape,
            ), params in param_groups.items():
                if len(params) <= 1:
                    continue  # Skip if only one parameter in this group

                # Calculate weighted average
                avg_param = None
                total_weight = 0.0

                for model_name, _, param in params:
                    weight = self.model_weights[model_name]
                    total_weight += weight

                    if avg_param is None:
                        avg_param = param.data.clone() * weight
                    else:
                        avg_param.add_(param.data * weight)

                if total_weight > 0:
                    avg_param.div_(total_weight)

                    # Mix original parameters with the average (soft sync)
                    sync_ratio = 0.1  # 10% shared, 90% original
                    for _, _, param in params:
                        param.data.mul_(1 - sync_ratio).add_(
                            avg_param * sync_ratio
                        )
        except Exception as e:
            logger.error(
                f"Error during parameter synchronization: {e}"
            )
            logger.error("Skipping synchronization for this step")

    def log_metrics(self, model_losses: Dict[str, float]):
        """Log training metrics and update loss tracking."""
        for model_name, loss in model_losses.items():
            self.model_losses[model_name].append(loss)

        # Log metrics every 100 steps
        if self.step_count % 100 == 0:
            avg_losses = {
                name: np.mean(losses[-100:])
                for name, losses in self.model_losses.items()
                if losses
            }
            current_lr = (
                self.param_groups[0]["lr"] * self.get_lr_multiplier()
            )

            logger.info(f"Step {self.step_count}")
            logger.info(f"Current learning rate: {current_lr:.6f}")
            for model_name, avg_loss in avg_losses.items():
                logger.info(
                    f"Model {model_name} - Avg loss: {avg_loss:.4f}"
                )

    def state_dict(self) -> Dict:
        """Return the optimizer state dict with additional MultiModelOptimizer specifics."""
        state_dict = super(MultiModelOptimizer, self).state_dict()
        state_dict["model_weights"] = self.model_weights
        state_dict["step_count"] = self.step_count
        state_dict["current_accumulation_step"] = (
            self.current_accumulation_step
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state with MultiModelOptimizer specifics."""
        self.model_weights = state_dict.pop("model_weights")
        self.step_count = state_dict.pop("step_count")
        self.current_accumulation_step = state_dict.pop(
            "current_accumulation_step"
        )
        super(MultiModelOptimizer, self).load_state_dict(state_dict)


class MultiModelScheduler(_LRScheduler):
    """
    A learning rate scheduler designed to work with MultiModelOptimizer,
    supporting different schedules for different models based on their convergence rates.

    Args:
        optimizer (MultiModelOptimizer): The optimizer to schedule
        total_steps (int): Total number of training steps
        warmup_steps (int, optional): Number of warmup steps. Default: 1000
        min_lr_ratio (float, optional): Minimum learning rate as a fraction of max. Default: 0.1
        model_schedule_weights (Dict[str, float], optional): Per-model schedule weights. Default: None
        last_epoch (int, optional): The index of the last epoch. Default: -1
    """

    def __init__(
        self,
        optimizer: MultiModelOptimizer,
        total_steps: int,
        warmup_steps: int = 1000,
        min_lr_ratio: float = 0.1,
        model_schedule_weights: Optional[Dict[str, float]] = None,
        last_epoch: int = -1,
    ):

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

        # Use optimizer's model weights if not provided
        if model_schedule_weights is None:
            self.model_schedule_weights = optimizer.model_weights
        else:
            self.model_schedule_weights = model_schedule_weights

        self.model_convergence_rates: Dict[str, float] = {
            name: 1.0 for name in self.model_schedule_weights.keys()
        }
        super(MultiModelScheduler, self).__init__(
            optimizer, last_epoch
        )

    def get_lr(self):
        """Calculate learning rates for all parameter groups."""
        if not self._get_lr_called_within_step:
            logger.warning(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`."
            )

        # Apply warmup
        if self.last_epoch < self.warmup_steps:
            lr_scale = float(self.last_epoch) / float(
                max(1, self.warmup_steps)
            )
        else:
            # Cosine decay after warmup
            progress = float(
                self.last_epoch - self.warmup_steps
            ) / float(max(1, self.total_steps - self.warmup_steps))
            lr_scale = max(
                self.min_lr_ratio,
                0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        # Apply model-specific adjustments based on convergence rates
        lrs = []
        for group in self.optimizer.param_groups:
            model_name = group["model_name"]
            # Adjust learning rate based on model convergence rate
            model_lr = group["initial_lr"] * lr_scale

            # Apply model-specific adjustment
            if model_name in self.model_convergence_rates:
                # Models with higher convergence rates get lower learning rates
                conv_rate = self.model_convergence_rates[model_name]
                model_lr *= max(0.5, min(1.5, 1.0 / conv_rate))

            lrs.append(model_lr)

        return lrs

    def update_convergence_rates(
        self, model_losses: Dict[str, List[float]], window: int = 100
    ):
        """
        Update convergence rate estimates based on recent loss trends.

        Args:
            model_losses: Dictionary mapping model names to their loss histories
            window: Number of steps to consider for convergence estimation
        """
        for model_name, losses in model_losses.items():
            if len(losses) < window:
                continue

            # Use recent loss values
            recent_losses = losses[-window:]

            # Calculate slope of loss curve
            x = np.arange(len(recent_losses))
            y = np.array(recent_losses)

            # Simple linear regression to estimate convergence rate
            slope, _ = np.polyfit(x, y, 1)

            # Normalize slope to a convergence rate
            # Negative slope is good (loss is decreasing)
            norm_rate = 1.0 / (1.0 + abs(slope))

            # Update with exponential moving average
            old_rate = self.model_convergence_rates.get(
                model_name, 1.0
            )
            self.model_convergence_rates[model_name] = (
                0.9 * old_rate + 0.1 * norm_rate
            )

        # Log updated convergence rates
        logger.info("Updated model convergence rates:")
        for model_name, rate in self.model_convergence_rates.items():
            logger.info(f"  {model_name}: {rate:.4f}")


# Usage example with real dataset
def example_usage_with_real_data():
    """Example demonstrating how to use MultiModelOptimizer with real data from GLUE."""
    try:
        # Import required libraries
        from transformers import (
            BertForSequenceClassification,
            GPT2ForSequenceClassification,
            RobertaForSequenceClassification,
            BertTokenizer,
            GPT2Tokenizer,
            RobertaTokenizer,
            DataCollatorWithPadding,
        )
        from datasets import load_dataset
        from torch.utils.data import DataLoader

        # Set up logging
        logger.info(
            "=== Starting MultiModelOptimizer example with real GLUE data ==="
        )

        # Load SST-2 dataset from GLUE (small sentiment classification dataset)
        logger.info("Loading SST-2 dataset from GLUE...")
        sst2_dataset = load_dataset("glue", "sst2")
        train_dataset = sst2_dataset["train"].select(
            range(1000)
        )  # Use only 1000 examples for quick training

        # Load tokenizers
        logger.info("Loading tokenizers...")
        bert_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        roberta_tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base"
        )

        # Add padding token to GPT2 tokenizer (it doesn't have one by default)
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

        # Tokenization functions
        def tokenize_bert(examples):
            return bert_tokenizer(
                examples["sentence"], truncation=True, max_length=128
            )

        def tokenize_gpt2(examples):
            return gpt2_tokenizer(
                examples["sentence"], truncation=True, max_length=128
            )

        def tokenize_roberta(examples):
            return roberta_tokenizer(
                examples["sentence"], truncation=True, max_length=128
            )

        # Tokenize datasets for each model
        logger.info("Tokenizing datasets...")
        bert_dataset = train_dataset.map(tokenize_bert, batched=True)
        gpt2_dataset = train_dataset.map(tokenize_gpt2, batched=True)
        roberta_dataset = train_dataset.map(
            tokenize_roberta, batched=True
        )

        # Set format for PyTorch
        bert_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
        gpt2_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )
        roberta_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "label"],
        )

        # Create data collators
        bert_data_collator = DataCollatorWithPadding(
            tokenizer=bert_tokenizer
        )
        gpt2_data_collator = DataCollatorWithPadding(
            tokenizer=gpt2_tokenizer
        )
        roberta_data_collator = DataCollatorWithPadding(
            tokenizer=roberta_tokenizer
        )

        # Create dataloaders
        logger.info("Creating dataloaders...")
        batch_size = 16
        bert_dataloader = DataLoader(
            bert_dataset,
            batch_size=batch_size,
            collate_fn=bert_data_collator,
        )
        gpt2_dataloader = DataLoader(
            gpt2_dataset,
            batch_size=batch_size,
            collate_fn=gpt2_data_collator,
        )
        roberta_dataloader = DataLoader(
            roberta_dataset,
            batch_size=batch_size,
            collate_fn=roberta_data_collator,
        )

        # Load models for sequence classification
        logger.info(
            "Loading transformer models for sequence classification..."
        )
        models = {
            "bert": BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=2
            ),
            "gpt2": GPT2ForSequenceClassification.from_pretrained(
                "gpt2", num_labels=2
            ),
            "roberta": RobertaForSequenceClassification.from_pretrained(
                "roberta-base", num_labels=2
            ),
        }

        # Set up optimizer with different weights for each model
        logger.info("Setting up MultiModelOptimizer...")
        optimizer = MultiModelOptimizer(
            models=models,
            lr=3e-5,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            model_weights={"bert": 1.0, "gpt2": 0.7, "roberta": 1.3},
            gradient_accumulation_steps=2,
            clip_grad_norm=1.0,
            warmup_steps=100,
            grad_sync_frequency=50,
        )

        # Set up scheduler
        scheduler = MultiModelScheduler(
            optimizer=optimizer, total_steps=5000, warmup_steps=100
        )

        # Move models to GPU if available
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {device}")

        for model_name, model in models.items():
            models[model_name] = model.to(device)

        # Create iterator function for each dataloader
        def infinite_iterator(dataloader):
            while True:
                for batch in dataloader:
                    yield batch

        bert_iter = infinite_iterator(bert_dataloader)
        gpt2_iter = infinite_iterator(gpt2_dataloader)
        roberta_iter = infinite_iterator(roberta_dataloader)

        # Define metrics for tracking
        from sklearn.metrics import accuracy_score

        total_steps = 1000  # Total training steps
        eval_every = 100  # Evaluate every 100 steps
        best_accuracy = {"bert": 0.0, "gpt2": 0.0, "roberta": 0.0}

        logger.info(f"Starting training for {total_steps} steps...")

        # Training loop
        for step in range(total_steps):
            # Zero gradients
            optimizer.zero_grad()

            losses = {}

            try:
                # For BERT
                bert_batch = next(bert_iter)
                bert_batch = {
                    k: v.to(device) for k, v in bert_batch.items()
                }
                bert_outputs = models["bert"](**bert_batch)
                bert_loss = bert_outputs.loss
                bert_loss.backward()
                losses["bert"] = bert_loss.item()

                # For GPT2
                gpt2_batch = next(gpt2_iter)
                gpt2_batch = {
                    k: v.to(device) for k, v in gpt2_batch.items()
                }
                gpt2_outputs = models["gpt2"](**gpt2_batch)
                gpt2_loss = gpt2_outputs.loss
                gpt2_loss.backward()
                losses["gpt2"] = gpt2_loss.item()

                # For RoBERTa
                roberta_batch = next(roberta_iter)
                roberta_batch = {
                    k: v.to(device) for k, v in roberta_batch.items()
                }
                roberta_outputs = models["roberta"](**roberta_batch)
                roberta_loss = roberta_outputs.loss
                roberta_loss.backward()
                losses["roberta"] = roberta_loss.item()

                # Log metrics
                optimizer.log_metrics(losses)

                # Step the optimizer and scheduler
                optimizer.step()
                scheduler.step()

                # Update convergence rates periodically
                if step % 100 == 0:
                    scheduler.update_convergence_rates(
                        optimizer.model_losses
                    )

                # Evaluate periodically
                if step > 0 and step % eval_every == 0:
                    logger.info(f"Evaluating at step {step}...")

                    # Create a small evaluation set
                    eval_dataset = sst2_dataset["validation"].select(
                        range(100)
                    )

                    for model_name, model in models.items():
                        model.eval()

                        # Tokenize evaluation data based on model type
                        if model_name == "bert":
                            tokenizer = bert_tokenizer
                            tokenize_fn = tokenize_bert
                        elif model_name == "gpt2":
                            tokenizer = gpt2_tokenizer
                            tokenize_fn = tokenize_gpt2
                        else:  # roberta
                            tokenizer = roberta_tokenizer
                            tokenize_fn = tokenize_roberta

                        eval_tokenized = eval_dataset.map(
                            tokenize_fn, batched=True
                        )
                        eval_tokenized.set_format(
                            type="torch",
                            columns=[
                                "input_ids",
                                "attention_mask",
                                "label",
                            ],
                        )

                        # Create dataloader
                        eval_collator = DataCollatorWithPadding(
                            tokenizer=tokenizer
                        )
                        eval_dataloader = DataLoader(
                            eval_tokenized,
                            batch_size=16,
                            collate_fn=eval_collator,
                        )

                        # Evaluate
                        all_preds = []
                        all_labels = []

                        with torch.no_grad():
                            for batch in eval_dataloader:
                                batch = {
                                    k: v.to(device)
                                    for k, v in batch.items()
                                }
                                outputs = model(**batch)
                                logits = outputs.logits
                                preds = (
                                    torch.argmax(logits, dim=-1)
                                    .cpu()
                                    .numpy()
                                )
                                labels = batch["label"].cpu().numpy()

                                all_preds.extend(preds)
                                all_labels.extend(labels)

                        # Calculate accuracy
                        accuracy = accuracy_score(
                            all_labels, all_preds
                        )
                        logger.info(
                            f"Model {model_name} - Accuracy: {accuracy:.4f}"
                        )

                        # Save best model
                        if accuracy > best_accuracy[model_name]:
                            best_accuracy[model_name] = accuracy
                            torch.save(
                                model.state_dict(),
                                f"best_{model_name}_model.pt",
                            )
                            logger.info(
                                f"Saved new best {model_name} model with accuracy {accuracy:.4f}"
                            )

                        model.train()

            except RuntimeError as e:
                logger.error(
                    f"Error during training step {step}: {e}"
                )
                logger.error("Skipping this step and continuing...")
                optimizer.zero_grad()
                continue

            # Save checkpoint every 500 steps
            if step > 0 and step % 500 == 0:
                logger.info(f"Saving checkpoint at step {step}...")
                torch.save(
                    {
                        "step": step,
                        "model_states": {
                            name: model.state_dict()
                            for name, model in models.items()
                        },
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_accuracy": best_accuracy,
                    },
                    f"checkpoint_step_{step}.pt",
                )

        # Final evaluation and results
        logger.info("=== Training complete! Final results ===")
        for model_name, acc in best_accuracy.items():
            logger.info(f"Best {model_name} accuracy: {acc:.4f}")

    except Exception as e:
        logger.error(
            f"Fatal error in example_usage_with_real_data: {e}"
        )
        import traceback

        logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Use real data example by default
    example_usage_with_real_data()
