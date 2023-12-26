import pytest
from io import StringIO
from contextlib import redirect_stdout
from swarms.utils.class_args_wrapper import print_class_parameters
from swarms.structs import Agent, Autoscaler
from fastapi import FastAPI
from fastapi.testclient import TestClient
from swarms.utils.class_args_wrapper import print_class_parameters
from swarms.structs import Agent, Autoscaler

app = FastAPI()


def test_print_class_parameters_agent():
    f = StringIO()
    with redirect_stdout(f):
        print_class_parameters(Agent)
    output = f.getvalue().strip()
    # Replace with the expected output for Agent class
    expected_output = (
        "Parameter: name, Type: <class 'str'>\nParameter: age, Type:"
        " <class 'int'>"
    )
    assert output == expected_output


def test_print_class_parameters_autoscaler():
    f = StringIO()
    with redirect_stdout(f):
        print_class_parameters(Autoscaler)
    output = f.getvalue().strip()
    # Replace with the expected output for Autoscaler class
    expected_output = (
        "Parameter: min_agents, Type: <class 'int'>\nParameter:"
        " max_agents, Type: <class 'int'>"
    )
    assert output == expected_output


def test_print_class_parameters_error():
    with pytest.raises(TypeError):
        print_class_parameters("Not a class")


@app.get("/parameters/{class_name}")
def get_parameters(class_name: str):
    classes = {"Agent": Agent, "Autoscaler": Autoscaler}
    if class_name in classes:
        return print_class_parameters(
            classes[class_name], api_format=True
        )
    else:
        return {"error": "Class not found"}


client = TestClient(app)


def test_get_parameters_agent():
    response = client.get("/parameters/Agent")
    assert response.status_code == 200
    # Replace with the expected output for Agent class
    expected_output = {"x": "<class 'int'>", "y": "<class 'int'>"}
    assert response.json() == expected_output


def test_get_parameters_autoscaler():
    response = client.get("/parameters/Autoscaler")
    assert response.status_code == 200
    # Replace with the expected output for Autoscaler class
    expected_output = {
        "min_agents": "<class 'int'>",
        "max_agents": "<class 'int'>",
    }
    assert response.json() == expected_output


def test_get_parameters_not_found():
    response = client.get("/parameters/NonexistentClass")
    assert response.status_code == 200
    assert response.json() == {"error": "Class not found"}
