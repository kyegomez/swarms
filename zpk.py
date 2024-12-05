from swarms import Agent
from loguru import logger
import random
import re

# Configure loguru
logger.add("zkp_log.log", rotation="500 KB", retention="10 days", level="INFO")


class ProverAgent:
    """
    Prover Agent for Zero Knowledge Proof.

    Responsibilities:
    - Generate commitments based on a secret.
    - Respond to challenges from the Verifier.

    Attributes:
        agent (Agent): Swarms agent instance.
        p (int): The prime modulus.
        g (int): The generator.
        x (int): The Prover's secret.
    """

    def __init__(self, p: int, g: int, secret: int):
        self.p = p
        self.g = g
        self.x = secret  # Prover's secret
        self.agent = Agent(
            agent_name="ProverAgent",
            model_name="gpt-4o-mini",
            max_loop=1,
            interactive=False,
            streaming_on=True,
            system_prompt=(
                "You are the Prover in a Zero Knowledge Proof (ZKP) system. "
                "Your responsibilities are to generate commitments based on a secret value and "
                "respond to challenges from the Verifier without revealing the secret. "
                "Follow mathematical rules of modular arithmetic when performing computations."
            ),
        )
        logger.info("Initialized ProverAgent with p={}, g={}, secret={}", p, g, secret)

    def generate_commitment(self) -> tuple[int, int]:
        """
        Generates a random commitment for the proof.

        Returns:
            tuple[int, int]: The random value (r) and the commitment (t).
        """
        r = random.randint(1, self.p - 2)
        task = (
            f"Compute the commitment t = g^r % p for g={self.g}, r={r}, p={self.p}. "
            "Return only the numerical value of t as an integer."
        )
        t = self.agent.run(task=task)
        t_value = self._extract_integer(t, "commitment")
        logger.info("Prover generated commitment: r={}, t={}", r, t_value)
        return r, t_value

    def _extract_integer(self, response: str, label: str) -> int:
        """
        Extracts an integer from the LLM response.

        Args:
            response (str): The response from the agent.
            label (str): A label for logging purposes.

        Returns:
            int: The extracted integer value.
        """
        try:
            # Use regex to find the first integer in the response
            match = re.search(r"\b\d+\b", response)
            if match:
                value = int(match.group(0))
                return value
            else:
                raise ValueError(f"No integer found in {label} response: {response}")
        except Exception as e:
            logger.error("Failed to extract integer from {label} response: {response}")
            raise ValueError(f"Invalid {label} response: {response}") from e

    def respond_to_challenge(self, r: int, c: int) -> int:
        """
        Computes the response to a challenge.

        Args:
            r (int): The random value used in the commitment.
            c (int): The challenge issued by the Verifier.

        Returns:
            int: The response (z).
        """
        task = f"Compute the response z = (r + c * x) % (p-1) for r={r}, c={c}, x={self.x}, p={self.p}."
        z = self.agent.run(task=task)
        logger.info("Prover responded to challenge: z={}", z)
        return int(z)


class VerifierAgent:
    """
    Verifier Agent for Zero Knowledge Proof.

    Responsibilities:
    - Issue challenges to the Prover.
    - Verify the Prover's response.

    Attributes:
        agent (Agent): Swarms agent instance.
        p (int): The prime modulus.
        g (int): The generator.
        y (int): The public value from the Prover.
    """

    def __init__(self, p: int, g: int, y: int):
        self.p = p
        self.g = g
        self.y = y  # Public value
        self.agent = Agent(
            agent_name="VerifierAgent",
            model_name="gpt-4o-mini",
            max_loop=1,
            interactive=False,
            streaming_on=True,
            system_prompt=(
                "You are the Verifier in a Zero Knowledge Proof (ZKP) system. "
                "Your responsibilities are to issue random challenges and verify the Prover's response. "
                "Use modular arithmetic to check if the proof satisfies g^z % p == (t * y^c) % p."
            ),
        )
        logger.info("Initialized VerifierAgent with p={}, g={}, y={}", p, g, y)

    def issue_challenge(self) -> int:
        """
        Issues a random challenge to the Prover.

        Returns:
            int: The challenge value (c).
        """
        c = random.randint(1, 10)
        logger.info("Verifier issued challenge: c={}", c)
        return c

    def verify_proof(self, t: int, z: int, c: int) -> bool:
        """
        Verifies the Prover's response.

        Args:
            t (int): The commitment from the Prover.
            z (int): The response from the Prover.
            c (int): The challenge issued to the Prover.

        Returns:
            bool: True if the proof is valid, False otherwise.
        """
        task = f"Verify if g^z % p == (t * y^c) % p for g={self.g}, z={z}, p={self.p}, t={t}, y={self.y}, c={c}."
        verification_result = self.agent.run(task=task)
        is_valid = verification_result.strip().lower() == "true"
        logger.info("Verifier checked proof: t={}, z={}, c={}, valid={}", t, z, c, is_valid)
        return is_valid


class CoordinatorAgent:
    """
    Coordinator for orchestrating the Zero Knowledge Proof protocol.

    Responsibilities:
    - Initialize parameters.
    - Facilitate interaction between Prover and Verifier agents.
    """

    def __init__(self, p: int, g: int, secret: int):
        self.p = p
        self.g = g
        self.prover = ProverAgent(p, g, secret)
        y = pow(g, secret, p)  # Public value
        self.verifier = VerifierAgent(p, g, y)
        logger.info("Coordinator initialized with p={}, g={}, secret={}", p, g, secret)

    def orchestrate(self) -> bool:
        """
        Orchestrates the Zero Knowledge Proof protocol.

        Returns:
            bool: True if the proof is valid, False otherwise.
        """
        logger.info("Starting ZKP protocol orchestration.")
        r, t = self.prover.generate_commitment()
        c = self.verifier.issue_challenge()
        z = self.prover.respond_to_challenge(r, c)
        is_valid = self.verifier.verify_proof(t, z, c)
        logger.info("ZKP protocol completed. Valid proof: {}", is_valid)
        return is_valid


if __name__ == "__main__":
    # Example parameters
    p = 23  # Prime number
    g = 5   # Generator
    secret = 7  # Prover's secret

    # Initialize the Coordinator and run the protocol
    coordinator = CoordinatorAgent(p, g, secret)
    result = coordinator.orchestrate()
    print(f"Zero Knowledge Proof Verification Result: {'Valid' if result else 'Invalid'}")
