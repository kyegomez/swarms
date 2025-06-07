from pydantic import BaseModel, Field
from typing import Optional

# from litellm.types import (
#     ChatCompletionPredictionContentParam,
# )


# class LLMCompletionRequest(BaseModel):
#     """Schema for LLM completion request parameters."""

#     model: Optional[str] = Field(
#         default=None,
#         description="The name of the language model to use for text completion",
#     )
#     temperature: Optional[float] = Field(
#         default=0.5,
#         description="Controls randomness of the output (0.0 to 1.0)",
#     )
#     top_p: Optional[float] = Field(
#         default=None,
#         description="Controls diversity via nucleus sampling",
#     )
#     n: Optional[int] = Field(
#         default=None, description="Number of completions to generate"
#     )
#     stream: Optional[bool] = Field(
#         default=None, description="Whether to stream the response"
#     )
#     stream_options: Optional[dict] = Field(
#         default=None, description="Options for streaming response"
#     )
#     stop: Optional[Any] = Field(
#         default=None,
#         description="Up to 4 sequences where the API will stop generating",
#     )
#     max_completion_tokens: Optional[int] = Field(
#         default=None,
#         description="Maximum tokens for completion including reasoning",
#     )
#     max_tokens: Optional[int] = Field(
#         default=None,
#         description="Maximum tokens in generated completion",
#     )
#     prediction: Optional[ChatCompletionPredictionContentParam] = (
#         Field(
#             default=None,
#             description="Configuration for predicted output",
#         )
#     )
#     presence_penalty: Optional[float] = Field(
#         default=None,
#         description="Penalizes new tokens based on existence in text",
#     )
#     frequency_penalty: Optional[float] = Field(
#         default=None,
#         description="Penalizes new tokens based on frequency in text",
#     )
#     logit_bias: Optional[dict] = Field(
#         default=None,
#         description="Modifies probability of specific tokens",
#     )
#     reasoning_effort: Optional[Literal["low", "medium", "high"]] = (
#         Field(
#             default=None,
#             description="Level of reasoning effort for the model",
#         )
#     )
#     seed: Optional[int] = Field(
#         default=None, description="Random seed for reproducibility"
#     )
#     tools: Optional[List] = Field(
#         default=None,
#         description="List of tools available to the model",
#     )
#     tool_choice: Optional[Union[str, dict]] = Field(
#         default=None, description="Choice of tool to use"
#     )
#     logprobs: Optional[bool] = Field(
#         default=None,
#         description="Whether to return log probabilities",
#     )
#     top_logprobs: Optional[int] = Field(
#         default=None,
#         description="Number of most likely tokens to return",
#     )
#     parallel_tool_calls: Optional[bool] = Field(
#         default=None,
#         description="Whether to allow parallel tool calls",
#     )

#     class Config:
#         allow_arbitrary_types = True


class ModelConfigOrigin(BaseModel):
    """Schema for model configuration origin."""

    model_url: Optional[str] = Field(
        default=None,
        description="The URL of the model to use for text completion",
    )

    api_key: Optional[str] = Field(
        default=None,
        description="The API key to use for the model",
    )

    class Config:
        allow_arbitrary_types = True
