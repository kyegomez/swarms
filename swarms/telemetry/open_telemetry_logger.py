from typing import Any, Dict, List, Optional, Union
import json
import requests
from loguru import logger
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode


class TelemetryProcessor:
    """
    A class to handle telemetry processing, including converting data to JSON,
    exporting it to an API server, and tracing the operations with OpenTelemetry.

    Attributes:
        service_name (str): The name of the service for tracing.
        otlp_endpoint (str): The endpoint URL for the OTLP exporter.
        tracer (Tracer): The tracer object used for creating spans.

    Methods:
        process_data(data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None) -> str:
            Converts input data to a JSON string.

        export_to_server(json_data: Optional[str] = None, api_url: Optional[str] = None) -> None:
            Sends the JSON data to the specified API server.
    """

    def __init__(
        self,
        service_name: str = "telemetry_service",
        otlp_endpoint: str = "http://localhost:4318/v1/traces",
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the TelemetryProcessor class with configurable settings.

        Args:
            service_name (str): The name of the service for tracing.
            otlp_endpoint (str): The endpoint URL for the OTLP exporter.
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint

        # Configure OpenTelemetry Tracing
        resource = Resource(
            attributes={SERVICE_NAME: self.service_name}, *args, **kwargs
        )
        trace.set_tracer_provider(
            TracerProvider(resource=resource), *args, **kwargs
        )
        self.tracer = trace.get_tracer(__name__)

        # Configure OTLP Exporter to send spans to a collector (e.g., Jaeger, Zipkin)
        otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        logger.debug(
            f"TelemetryProcessor initialized with service_name={self.service_name}, otlp_endpoint={self.otlp_endpoint}"
        )

    def process_data(
        self,
        data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> str:
        """
        Converts input data to a JSON string.

        Args:
            data (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]): The input data to be converted.
                Defaults to an empty dictionary if None is provided.

        Returns:
            str: The JSON string representation of the input data.

        Raises:
            TypeError: If the input data is not a dictionary or a list of dictionaries.
            json.JSONEncodeError: If the data cannot be serialized to JSON.
        """
        with self.tracer.start_as_current_span("process_data") as span:
            if data is None:
                data = {}
            logger.debug(f"Processing data: {data}")

            if not isinstance(data, (dict, list)):
                logger.error(
                    "Invalid data type. Expected a dictionary or a list of dictionaries."
                )
                span.set_status(
                    Status(StatusCode.ERROR, "Invalid data type")
                )
                raise TypeError(
                    "Input data must be a dictionary or a list of dictionaries."
                )

            try:
                json_data = json.dumps(data)
                logger.debug(f"Converted data to JSON: {json_data}")
                return json_data
            except (TypeError, json.JSONEncodeError) as e:
                logger.error(f"Failed to convert data to JSON: {e}")
                span.set_status(
                    Status(StatusCode.ERROR, "JSON serialization failed")
                )
                raise

    def export_to_server(
        self,
        json_data: Optional[str] = None,
        api_url: Optional[str] = None,
    ) -> None:
        """
        Sends the JSON data to the specified API server.

        Args:
            json_data (Optional[str]): The JSON data to be sent. Defaults to an empty JSON string if None is provided.
            api_url (Optional[str]): The URL of the API server to send the data to. Defaults to None.

        Raises:
            ValueError: If the api_url is None.
            requests.exceptions.RequestException: If there is an error sending the data to the server.
        """
        with self.tracer.start_as_current_span("export_to_server") as span:
            if json_data is None:
                json_data = "{}"
            if api_url is None:
                logger.error("API URL cannot be None.")
                span.set_status(
                    Status(StatusCode.ERROR, "API URL is missing")
                )
                raise ValueError("API URL cannot be None.")

            logger.debug(f"Exporting JSON data to server: {api_url}")
            headers = {"Content-Type": "application/json"}

            log = {
                "data": json_data,
            }

            try:
                response = requests.post(
                    api_url, data=log, headers=headers
                )
                response.raise_for_status()
                logger.info(
                    f"Data successfully exported to {api_url}: {response.status_code}"
                )
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to export data to {api_url}: {e}")
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        "Failed to send data to API server",
                    )
                )
                raise


# # Example usage:

# if __name__ == "__main__":
#     # Example usage with custom service name and OTLP endpoint
#     processor = TelemetryProcessor(service_name="my_telemetry_service", otlp_endpoint="http://my-collector:4318/v1/traces")

#     # Sample data
#     telemetry_data = {
#         "device_id": "sensor_01",
#         "temperature": 22.5,
#         "humidity": 60,
#         "timestamp": "2024-08-15T12:34:56Z"
#     }

#     # Processing data
#     try:
#         json_data = processor.process_data(telemetry_data)
#     except Exception as e:
#         logger.error(f"Processing error: {e}")
#         # Handle error accordingly

#     # Exporting data to an API server
#     api_url = "https://example.com/api/telemetry"
#     try:
#         processor.export_to_server(json_data, api_url)
#     except Exception as e:
#         logger.error(f"Export error: {e}")
#         # Handle error accordingly
