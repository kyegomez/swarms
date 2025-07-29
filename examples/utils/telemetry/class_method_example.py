"""
Example usage of log_function_execution decorator with class methods.

This demonstrates how the decorator works with:
- Instance methods
- Class methods
- Static methods
- Property methods
"""

from swarms.telemetry.log_executions import log_function_execution


class DataProcessor:
    """Example class to demonstrate decorator usage with methods."""

    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.processed_count = 0

    @log_function_execution(
        swarm_id="data-processor-instance",
        swarm_architecture="object_oriented",
        enabled_on=True,
    )
    def process_data(self, data: list, multiplier: int = 2) -> dict:
        """Instance method that processes data."""
        processed = [x * multiplier for x in data]
        self.processed_count += len(data)

        return {
            "original": data,
            "processed": processed,
            "processor_name": self.name,
            "count": len(processed),
        }

    @classmethod
    @log_function_execution(
        swarm_id="data-processor-class",
        swarm_architecture="class_method",
        enabled_on=True,
    )
    def create_default(cls, name: str):
        """Class method to create a default instance."""
        return cls(name=name, version="default")

    @staticmethod
    @log_function_execution(
        swarm_id="data-processor-static",
        swarm_architecture="utility",
        enabled_on=True,
    )
    def validate_data(data: list) -> bool:
        """Static method to validate data."""
        return isinstance(data, list) and len(data) > 0

    @property
    def status(self) -> str:
        """Property method (not decorated as it's a getter)."""
        return f"{self.name} v{self.version} - {self.processed_count} items processed"


class AdvancedProcessor(DataProcessor):
    """Subclass to test inheritance with decorated methods."""

    @log_function_execution(
        swarm_id="advanced-processor",
        swarm_architecture="inheritance",
        enabled_on=True,
    )
    def advanced_process(
        self, data: list, algorithm: str = "enhanced"
    ) -> dict:
        """Advanced processing method in subclass."""
        base_result = self.process_data(data, multiplier=3)

        return {
            **base_result,
            "algorithm": algorithm,
            "advanced": True,
            "processor_type": "AdvancedProcessor",
        }


if __name__ == "__main__":
    print("Testing decorator with class methods...")

    # Test instance method
    print("\n1. Testing instance method:")
    processor = DataProcessor("TestProcessor", "2.0")
    result1 = processor.process_data([1, 2, 3, 4], multiplier=5)
    print(f"Result: {result1}")
    print(f"Status: {processor.status}")

    # Test class method
    print("\n2. Testing class method:")
    default_processor = DataProcessor.create_default(
        "DefaultProcessor"
    )
    print(
        f"Created: {default_processor.name} v{default_processor.version}"
    )
