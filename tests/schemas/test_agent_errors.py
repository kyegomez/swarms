from concurrent.futures import ProcessPoolExecutor

import pytest
from pydantic import BaseModel, ValidationError

from swarms.schemas.agent_errors import AgentOutputSchemaError


class Report(BaseModel):
    score: int


RAW_OUTPUT = '{"score": "high"}'


def raise_output_schema_error(raw: str) -> None:
    try:
        Report.model_validate_json(raw)
    except ValidationError as error:
        raise AgentOutputSchemaError(
            "Agent did not return a valid Report",
            raw_output=raw,
            validation_error=error,
        ) from error


def test_output_schema_error_reaches_a_process_pool_caller():
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(raise_output_schema_error, RAW_OUTPUT)
        with pytest.raises(AgentOutputSchemaError) as raised:
            future.result(timeout=60)

    assert str(raised.value) == "Agent did not return a valid Report"
    assert raised.value.raw_output == RAW_OUTPUT
    assert "score" in str(raised.value.validation_error)
