"""
Council of Judges Examples Package.

This package contains examples demonstrating how to use the CouncilAsAJudge
class for evaluating task responses across multiple dimensions.
"""

from .council_judge_example import main as basic_example
from .council_judge_complex_example import main as complex_example
from .council_judge_custom_example import main as custom_example

__all__ = ["basic_example", "complex_example", "custom_example"]
