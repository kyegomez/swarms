import json
from typing import Any, List

from loguru import logger
from pydantic import BaseModel, Field

from swarms import Agent


class AgentOutput(BaseModel):
    """
    Schema for capturing metadata and results of an agent run.
    """

    agent_name: str = Field(..., description="Name of the agent.")
    input_query: str = Field(
        ..., description="Input query provided to the agent."
    )
    output_result: Any = Field(
        ..., description="Result produced by the agent."
    )
    metadata: dict = Field(
        ..., description="Additional metadata about the agent run."
    )


class MatrixSwarm:
    """
    A class to manage a matrix of agents and perform matrix operations similar to linear algebra.
    """

    def __init__(self, agents: List[List[Agent]]):
        """
        Initializes the MatrixSwarm with a 2D list of agents.
        Args:
            agents (List[List[Agent]]): 2D list of agents representing the matrix.
        """
        if not agents or not all(
            isinstance(row, list) for row in agents
        ):
            raise ValueError("Agents must be provided as a 2D list.")
        if not all(
            isinstance(agent, Agent)
            for row in agents
            for agent in row
        ):
            raise ValueError(
                "All elements of the matrix must be instances of `Agent`."
            )
        self.agents = agents
        self.outputs = []  # List to store outputs as AgentOutput

    def validate_dimensions(self, other: "MatrixSwarm") -> None:
        """
        Validates that two matrices have compatible dimensions for operations.

        Args:
            other (MatrixSwarm): Another MatrixSwarm.

        Raises:
            ValueError: If dimensions are incompatible.
        """
        if len(self.agents) != len(other.agents) or len(
            self.agents[0]
        ) != len(other.agents[0]):
            raise ValueError(
                "Matrix dimensions are incompatible for this operation."
            )

    def transpose(self) -> "MatrixSwarm":
        """
        Transposes the matrix of agents (swap rows and columns).

        Returns:
            MatrixSwarm: A new transposed MatrixSwarm.
        """
        transposed_agents = [
            [self.agents[j][i] for j in range(len(self.agents))]
            for i in range(len(self.agents[0]))
        ]
        return MatrixSwarm(transposed_agents)

    def add(self, other: "MatrixSwarm") -> "MatrixSwarm":
        """
        Adds two matrices element-wise.

        Args:
            other (MatrixSwarm): Another MatrixSwarm to add.

        Returns:
            MatrixSwarm: A new MatrixSwarm resulting from the addition.
        """
        self.validate_dimensions(other)
        added_agents = [
            [self.agents[i][j] for j in range(len(self.agents[i]))]
            for i in range(len(self.agents))
        ]
        return MatrixSwarm(added_agents)

    def scalar_multiply(self, scalar: int) -> "MatrixSwarm":
        """
        Scales the agents by duplicating them scalar times along the row.

        Args:
            scalar (int): The scalar multiplier.

        Returns:
            MatrixSwarm: A new MatrixSwarm where each agent is repeated scalar times along the row.
        """
        scaled_agents = [
            [agent for _ in range(scalar) for agent in row]
            for row in self.agents
        ]
        return MatrixSwarm(scaled_agents)

    def multiply(
        self, other: "MatrixSwarm", inputs: List[str]
    ) -> List[List[AgentOutput]]:
        """
        Multiplies two matrices (dot product between rows and columns).

        Args:
            other (MatrixSwarm): Another MatrixSwarm for multiplication.
            inputs (List[str]): A list of input queries for the agents.

        Returns:
            List[List[AgentOutput]]: A resulting matrix of outputs after multiplication.
        """
        if len(self.agents[0]) != len(other.agents):
            raise ValueError(
                "Matrix dimensions are incompatible for multiplication."
            )

        results = []
        for i, row in enumerate(self.agents):
            row_results = []
            for col_idx in range(len(other.agents[0])):
                col = [
                    other.agents[row_idx][col_idx]
                    for row_idx in range(len(other.agents))
                ]
                query = inputs[
                    i
                ]  # Input query for the corresponding row
                intermediate_result = []

                for agent_r, agent_c in zip(row, col):
                    try:
                        result = agent_r.run(query)
                        intermediate_result.append(result)
                    except Exception as e:
                        intermediate_result.append(f"Error: {e}")

                # Aggregate outputs from dot product
                combined_result = " ".join(
                    intermediate_result
                )  # Example aggregation
                row_results.append(
                    AgentOutput(
                        agent_name=f"DotProduct-{i}-{col_idx}",
                        input_query=query,
                        output_result=combined_result,
                        metadata={"row": i, "col": col_idx},
                    )
                )
            results.append(row_results)
        return results

    def subtract(self, other: "MatrixSwarm") -> "MatrixSwarm":
        """
        Subtracts two matrices element-wise.

        Args:
            other (MatrixSwarm): Another MatrixSwarm to subtract.

        Returns:
            MatrixSwarm: A new MatrixSwarm resulting from the subtraction.
        """
        self.validate_dimensions(other)
        subtracted_agents = [
            [self.agents[i][j] for j in range(len(self.agents[i]))]
            for i in range(len(self.agents))
        ]
        return MatrixSwarm(subtracted_agents)

    def identity(self, size: int) -> "MatrixSwarm":
        """
        Creates an identity matrix of agents with size `size`.

        Args:
            size (int): Size of the identity matrix (NxN).

        Returns:
            MatrixSwarm: An identity MatrixSwarm.
        """
        identity_agents = [
            [
                (
                    self.agents[i][j]
                    if i == j
                    else Agent(
                        agent_name=f"Zero-Agent-{i}-{j}",
                        system_prompt="",
                    )
                )
                for j in range(size)
            ]
            for i in range(size)
        ]
        return MatrixSwarm(identity_agents)

    def determinant(self) -> Any:
        """
        Computes the determinant of a square MatrixSwarm.

        Returns:
            Any: Determinant of the matrix (as agent outputs).
        """
        if len(self.agents) != len(self.agents[0]):
            raise ValueError(
                "Determinant can only be computed for square matrices."
            )

        # Recursive determinant calculation (example using placeholder logic)
        if len(self.agents) == 1:
            return self.agents[0][0].run("Compute determinant")

        det_result = 0
        for i in range(len(self.agents)):
            submatrix = MatrixSwarm(
                [row[:i] + row[i + 1 :] for row in self.agents[1:]]
            )
            cofactor = ((-1) ** i) * self.agents[0][i].run(
                "Compute determinant"
            )
            det_result += cofactor * submatrix.determinant()
        return det_result

    def save_to_file(self, path: str) -> None:
        """
        Saves the agent matrix structure and metadata to a file.

        Args:
            path (str): File path to save the matrix.
        """
        try:
            matrix_data = {
                "agents": [
                    [agent.agent_name for agent in row]
                    for row in self.agents
                ],
                "outputs": [output.dict() for output in self.outputs],
            }
            with open(path, "w") as f:
                json.dump(matrix_data, f, indent=4)
            logger.info(f"MatrixSwarm saved to {path}")
        except Exception as e:
            logger.error(f"Error saving MatrixSwarm: {e}")


# # Example usage
# if __name__ == "__main__":
#     from swarms.prompts.finance_agent_sys_prompt import (
#         FINANCIAL_AGENT_SYS_PROMPT,
#     )

#     # Create a 3x3 matrix of agents
#     agents = [
#         [
#             Agent(
#                 agent_name=f"Agent-{i}-{j}",
#                 system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#                 model_name="gpt-4o-mini",
#                 max_loops=1,
#                 autosave=True,
#                 dashboard=False,
#                 verbose=True,
#                 dynamic_temperature_enabled=True,
#                 saved_state_path=f"agent_{i}_{j}.json",
#                 user_name="swarms_corp",
#                 retry_attempts=1,
#                 context_length=200000,
#                 return_step_meta=False,
#                 output_type="string",
#                 streaming_on=False,
#             )
#             for j in range(3)
#         ]
#         for i in range(3)
#     ]

#     # Initialize the matrix
#     agent_matrix = MatrixSwarm(agents)

#     # Example queries
#     inputs = [
#         "Explain Roth IRA benefits",
#         "Differences between ETFs and mutual funds",
#         "How to create a diversified portfolio",
#     ]

#     # Run agents
#     outputs = agent_matrix.multiply(agent_matrix.transpose(), inputs)

#     # Save results
#     agent_matrix.save_to_file("agent_matrix_results.json")
