import pickle

from pydantic import BaseModel, ValidationError

from swarms.schemas.agent_errors import AgentOutputSchemaError


class Report(BaseModel):
    score: int


def test_output_schema_error_survives_pickle_round_trip():
    raw_output = '{"score": "high"}'

    try:
        Report.model_validate_json(raw_output)
    except ValidationError as error:
        schema_error = AgentOutputSchemaError(
            "Agent did not return a valid Report",
            raw_output=raw_output,
            validation_error=error,
        )

    restored = pickle.loads(pickle.dumps(schema_error))

    assert type(restored) is AgentOutputSchemaError
    assert str(restored) == "Agent did not return a valid Report"
    assert restored.raw_output == raw_output
    assert "score" in str(restored.validation_error)
