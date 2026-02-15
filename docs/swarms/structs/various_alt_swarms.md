# Various Alt Swarms

    `various_alt_swarms` reference documentation.

    **Module Path**: `swarms.structs.various_alt_swarms`

    ## Overview

    Collection of alternative swarm/communication patterns (circular, star, mesh, pyramid, broadcast, one-to-one, etc.).

    ## Public API

    - **`BaseSwarm`**: `run()`
- **`CircularSwarm`**: `run()`
- **`StarSwarm`**: `run()`
- **`MeshSwarm`**: `run()`
- **`PyramidSwarm`**: `run()`
- **`FibonacciSwarm`**: `run()`
- **`PrimeSwarm`**: `run()`
- **`PowerSwarm`**: `run()`
- **`LogSwarm`**: `run()`
- **`ExponentialSwarm`**: `run()`
- **`GeometricSwarm`**: `run()`
- **`HarmonicSwarm`**: `run()`
- **`StaircaseSwarm`**: `run()`
- **`SigmoidSwarm`**: `run()`
- **`SinusoidalSwarm`**: `run()`
- **`OneToOne`**: `run()`
- **`Broadcast`**: `run()`
- **`OneToThree`**: `run()`

    ## Quickstart

    ```python
    from swarms.structs.various_alt_swarms import BaseSwarm, CircularSwarm, StarSwarm
    ```

    ## Tutorial

    A runnable tutorial is available at [`swarms/examples/various_alt_swarms_example.md`](../examples/various_alt_swarms_example.md).

    ## Notes

    - Keep task payloads small for first runs.
    - Prefer deterministic prompts when comparing outputs across agents.
    - Validate provider credentials (for LLM-backed examples) before production use.
