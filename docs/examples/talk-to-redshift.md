This example demonstrates how to build an agent that can dynamically query Amazon Redshift Serverless tables and store its contents on the local hard drive.

Let's build a support agent that uses GPT-4:

```python
import boto3
from swarms.drivers import AmazonRedshiftSqlDriver, OpenAiPromptDriver
from swarms.loaders import SqlLoader
from swarms.rules import Ruleset, Rule
from swarms.structures import Agent
from swarms.tools import SqlClient, FileManager
from swarms.utils import Chat

session = boto3.Session(region_name="REGION_NAME")

sql_loader = SqlLoader(
    sql_driver=AmazonRedshiftSqlDriver(
        database="DATABASE",
        session=session,
        workgroup_name="WORKGROUP_NAME"
    )
)

sql_tool = SqlClient(
    sql_loader=sql_loader,
    table_name="people",
    table_description="contains information about tech industry professionals",
    engine_name="redshift"
)

agent = Agent(
    tools=[sql_tool, FileManager())],
    rulesets=[
        Ruleset(
            name="HumansOrg Agent",
            rules=[
                Rule("Act and introduce yourself as a HumansOrg, Inc. support agent"),
                Rule("Your main objective is to help with finding information about people"),
                Rule("Only use information about people from the sources available to you")
            ]
        )
    ]
)

Chat(agent).start()
```
