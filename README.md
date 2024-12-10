<div align="center">
  <a href="https://swarms.world">
    <img src="https://github.com/kyegomez/swarms/blob/master/images/swarmslogobanner.png" style="margin: 15px; max-width: 300px" width="50%" alt="Logo">
  </a>
</div>
<p align="center">
  <em>The Enterprise-Grade Production-Ready Multi-Agent Orchestration Framework </em>
</p>

<p align="center">
    <a href="https://pypi.org/project/swarms/" target="_blank">
        <img alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
        <img alt="Version" src="https://img.shields.io/pypi/v/swarms?style=for-the-badge&color=3670A0">
    </a>
</p>
<p align="center">
<a href="https://twitter.com/swarms_corp/">🐦 Twitter</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://discord.gg/agora-999382051935506503">📢 Discord</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://swarms.world">Swarms Platform</a>
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://docs.swarms.world">📙 Documentation</a>
</p>

[Previous content remains unchanged until the new section...]

## `AsyncWorkflow`

The `AsyncWorkflow` class enables asynchronous execution of multiple agents in parallel, allowing for efficient task processing and improved performance. This workflow is particularly useful when dealing with multiple independent tasks that can be executed concurrently.

### Methods

| Method | Description | Parameters | Return Value |
|--------|-------------|------------|--------------|
| `__init__` | Initialize the AsyncWorkflow | `agents`: List of Agent objects<br>`max_workers`: Maximum number of concurrent workers | None |
| `run` | Execute tasks asynchronously | `tasks`: List of tasks to be processed | List of results from all tasks |

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| `agents` | List[Agent] | List of Agent objects to process tasks |
| `max_workers` | int | Maximum number of concurrent worker threads |
| `tasks` | List[str] | List of tasks to be processed |

### Output

The `run` method returns a list containing the results from all processed tasks.

### Example Usage
