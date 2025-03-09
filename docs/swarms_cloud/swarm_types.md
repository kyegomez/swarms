### Available Swarms in The Swarms API

| Swarm Type           | Description (English)                                                      | Description (Chinese)                                                      |
|----------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------|
| AgentRearrange       | A swarm type focused on rearranging agents for optimal performance.         | 一种专注于重新排列代理以实现最佳性能的群类型。                               |
| MixtureOfAgents      | Combines different types of agents to achieve a specific goal.              | 结合不同类型的代理以实现特定目标。                                         |
| SpreadSheetSwarm     | Utilizes spreadsheet-like structures for data management and operations.    | 利用类似电子表格的结构进行数据管理和操作。                                 |
| SequentialWorkflow   | Executes tasks in a sequential manner.                                      | 以顺序方式执行任务。                                                       |
| ConcurrentWorkflow   | Allows tasks to be executed concurrently for efficiency.                    | 允许任务并发执行以提高效率。                                               |
| GroupChat            | Facilitates communication among agents in a group chat format.             | 以群聊格式促进代理之间的沟通。                                             |
| MultiAgentRouter     | Routes tasks and information among multiple agents.                         | 在多个代理之间路由任务和信息。                                             |
| AutoSwarmBuilder     | Automatically builds and configures swarms based on predefined criteria.    | 根据预定义标准自动构建和配置群。                                           |
| HiearchicalSwarm     | Organizes agents in a hierarchical structure for task delegation.           | 以层次结构组织代理以进行任务委派。                                         |
| auto                 | Automatically selects the best swarm type based on the context.             | 根据上下文自动选择最佳群类型。                                             |
| MajorityVoting       | Uses majority voting among agents to make decisions.                        | 使用代理之间的多数投票来做出决策。                                         |
| MALT                 | A specialized swarm type for specific tasks (details needed).               | 一种专门为特定任务设计的群类型（需要详细信息）。                           |

### Documentation for Swarms

1. **AgentRearrange**: This swarm type is designed to rearrange agents to optimize their performance in a given task. It is useful in scenarios where agent positioning or order affects the outcome.
   - 这种群类型旨在重新排列代理以优化其在给定任务中的性能。它在代理位置或顺序影响结果的情况下非常有用。

2. **MixtureOfAgents**: This type combines various agents, each with unique capabilities, to work together towards a common goal. It leverages the strengths of different agents to enhance overall performance.
   - 这种类型结合了各种代理，每个代理都有独特的能力，共同努力实现共同目标。它利用不同代理的优势来提高整体性能。

3. **SpreadSheetSwarm**: This swarm type uses spreadsheet-like structures to manage and operate on data. It is ideal for tasks that require organized data manipulation and analysis.
   - 这种群类型使用类似电子表格的结构来管理和操作数据。它非常适合需要有组织的数据操作和分析的任务。

4. **SequentialWorkflow**: Tasks are executed one after another in this swarm type, ensuring that each step is completed before the next begins. It is suitable for processes that require strict order.
   - 在这种群类型中，任务一个接一个地执行，确保每个步骤在下一个步骤开始之前完成。它适用于需要严格顺序的流程。

5. **ConcurrentWorkflow**: This type allows multiple tasks to be executed simultaneously, improving efficiency and reducing time for completion. It is best for independent tasks that do not rely on each other.
   - 这种类型允许多个任务同时执行，提高效率并减少完成时间。它最适合不相互依赖的独立任务。

6. **GroupChat**: Facilitates communication among agents in a group chat format, enabling real-time collaboration and decision-making.
   - 以群聊格式促进代理之间的沟通，实现实时协作和决策。

7. **MultiAgentRouter**: This swarm type routes tasks and information among multiple agents, ensuring that each agent receives the necessary data to perform its function.
   - 这种群类型在多个代理之间路由任务和信息，确保每个代理接收到执行其功能所需的数据。

8. **AutoSwarmBuilder**: Automatically builds and configures swarms based on predefined criteria, reducing the need for manual setup and configuration.
   - 根据预定义标准自动构建和配置群，减少手动设置和配置的需要。

9. **HiearchicalSwarm**: Organizes agents in a hierarchical structure, allowing for efficient task delegation and management.
   - 以层次结构组织代理，允许高效的任务委派和管理。

10. **auto**: Automatically selects the most appropriate swarm type based on the context and requirements of the task.
    - 根据任务的上下文和要求自动选择最合适的群类型。

11. **MajorityVoting**: Uses a majority voting mechanism among agents to make decisions, ensuring that the most popular choice is selected.
    - 使用代理之间的多数投票机制来做出决策，确保选择最受欢迎的选项。

12. **MALT**: A specialized swarm type designed for specific tasks. Further details are needed to fully document this type.
    - 一种专门为特定任务设计的群类型。需要进一步的详细信息来完整记录这种类型。
