from swarms.tools.agent.singletool import load_single_tools, STQuestionAnswerer

tool_name, tool_url = "database", "http://127.0.0.1:8079/tools/database/"
tool_name, tool_config = load_single_tools(tool_name, tool_url)
print(tool_name, tool_config)
stqa = STQuestionAnswerer()

agent = stqa.load_tools(
    tool_name, tool_config, prompt_type="autogpt"
)  # langchain: react-with-tool-description

# 394
text = "Retrieve the comments of suppliers, size of parts, and supply cost of part-supplier combinations where the available quantity of the part is 6331, the type of the part is greater than 'LARGE POLISHED NICKEL', and the retail price of the part is less than 1758.76. The results should be sorted in descending order based on the comments of the suppliers."
# 1707
# text = "Retrieve the tax rate and total price from the lineitem and orders tables where the line number is greater than or equal to 3, the order key is not equal to 784709, the order key is less than or equal to 189383, and the clerk's ID is less than 'Clerk#000000181'."
# # 100967
# text = "Retrieve the part type, available quantity of parts, and the sum of supplier keys from the part and partsupp tables where the supplier key is not equal to 3804, the part key is less than or equal to 57823, and the available quantity of parts is less than 4781. Group the results by part type and available quantity of parts, and only include groups where the sum of supplier keys is greater than 1089. Sort the results in ascending order based on the sum of supplier keys."
# # 1249285 (fail to generate the sql)
# text = "Retrieve the phone number of the customer, the total price of the order, the comment of the nation, and the comment of the region from the orders table, customer table, nation table, and region table, respectively, where the nation key is less than 8, the order status is greater than or equal to 'O', and the order comment is less than 'ly around the pending theodo'. Sort the result by customer phone number in ascending order, nation comment in ascending order, order total price in ascending order, and region comment in ascending order."
# # 1272240
# text = "Retrieve the account balance, supply cost, region key, and nation name from the region, nation, supplier, and partsupp tables where the region name is less than or equal to 'AFRICA', the nation comment is greater than or equal to 'l platelets. regular accounts x-ray: unusual, regular acco', and the supplier nation key is greater than or equal to 0."
# # 150302
# text = "Retrieve the order key and customer address from the customer and orders tables where the customer phone number is not '29-716-678-7355', the customer key is less than or equal to 16201, the order total price is greater than 29849.7, the order clerk is not 'Clerk#000000361', and the order ship priority is greater than or equal to 0. Sort the results by customer address in descending order."
# 3376197
# text = "Retrieve the shipment dates from the lineitem table where the extended price is greater than or equal to 50883.12, the linenumber is greater than 1, the shipdate is not equal to '1992-08-30', and the return flag is 'A', and sort the results in descending order based on the shipment date." # 7
# 7
# text = "Retrieve the account balance, order priority, and nation name for customers who have a comment of 'ar deposits believe special, express foxes. packages cajole slyly e', are not from Japan, have a market segment of 'HOUSEHOLD', have a total order price less than 110238.65, and have a name less than or equal to 'Customer#000013191'."   # 8
# 974546
# text = "Retrieve the part type and part supplier comment from the Part and Partsupp tables where the available quantity of the part supplier is not equal to 1078, the part type is less than 'PROMO BURNISHED NICKEL', the part size is greater than 8, and the part container is less than 'LG CAN'." # 9
# 85446
# text = "Retrieve the comments from the \"partsupp\" table where the available quantity is greater than or equal to 9324, the supplier key is not equal to 1716, the part key is greater than or equal to 65143, the supply cost is less than 164.19, and the comment is not equal to 's use slyly pending instructions. furiously final ideas shall have to are c'." # 10
# 623025 (wrong results ~ directly call the database tool)
# text = "Retrieve the supplier name, supplier key, and region key from the supplier, nation, and region tables where the region key is greater than or equal to 1, the supplier key is less than or equal to 9696, the region comment is not equal to 'uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl', the supplier name is less than 'Supplier#000008309', and the supplier phone is not equal to '19-247-536-8083', and sort the results by supplier key in ascending order, region key in descending order, and region name in ascending order."  # 11
# 14448 (wrong results)
# text = "Retrieve the order priority from the \"orders\" table where the order priority is greater than '3-MEDIUM', the total price is greater than 130861.55, the comment is less than 'finally pending packages sleep along the furiously special', the customer key is less than or equal to 16480, the ship priority is less than or equal to 0, and the order date is not equal to '1997-02-20', and sort the results in ascending order based on the order priority." # 12

# rewrite
# text = "SELECT s_comment FROM part As p,partsupp As ps,supplier As s WHERE p.p_partkey = ps.ps_partkey AND s.s_suppkey = ps.ps_suppkey AND ps.ps_availqty = 6331 AND p.p_type > 'LARGE POLISHED NICKEL' AND p.p_retailprice < 1758.76 ORDER BY s_comment DESC;"
# text = "Retrieve the comments of suppliers. The results should be sorted in descending order based on the comments of the suppliers"
text = "Retrieve the comments in the supplier table where the p\_partkey column in the part table matches the ps\_partkey column in the partsupp table, the ps\_availqty column in the partsupp table equals 6331, the p_type column in the part table is greater than 'LARGE POLISHED NICKEL', and the p\_retailprice column in the part table is less than 1758.76."

agent.run(
    [
        """ First get the database schema via get_database_schema. Next generate the sql query exactly based on the schema and the following description:
\"{}\" 

Next rewrite the SQL query and output the total number of rows in the database results of the rewritten SQL query.

Note. 1) Only obtain the database schema once;
2) If an API is successfully called, do not call the same API again; 
3) Do not use any image in the output; 
4) The db_name is tpch10x; 
5) Count the rows of query results by your own and do not output the whole query results.
""".format(
            text
        )
    ]
)

# # unit test: get_database_schema
# agent.run(["""
# Fetch the database schema from a postgresql database named tpch10x.\"
# """])


# # unit test: rewrite_sql
# agent("Rewrite the input query: select * from customer limit 2")


# # unit test: select_database_data
# agent("""
# Output the total number of rows in the query results from a postgresql database based on the following description:

# \"Retrieve all the data from the 'customer' table and limit the output to only the first 2 rows.\"
# """)

""" output (autogpt)
> Entering new LLMChain chain...
Prompt after formatting:
System: You are Tom, Assistant
Your decisions must always be made independently 
            without seeking user assistance. Play to your strengths 
            as an LLM and pursue simple strategies with no legal complications. 
            If you have completed all your tasks, 
            make sure to use the "finish" command.

GOALS:

{input prompt}

Constraints:
1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. No user assistance
4. Exclusively use the commands listed in double quotes e.g. "command name"

Commands:
1. get_database_schema: . Your input should be a json (args json schema): {{"db_name" : string, }} The Action to trigger this API should be get_database_schema and the input parameters should be a json dict string. Pay attention to the type of parameters.
2. select_database_data: Read the data stored in database. Your input should be a json (args json schema): {{"query" : string, }} The Action to trigger this API should be select_database_data and the input parameters should be a json dict string. Pay attention to the type of parameters.
3. rewrite_sql: Get rewritten sql from rewriter. Your input should be a json (args json schema): {{"sql" : string, }} The Action to trigger this API should be rewrite_sql and the input parameters should be a json dict string. Pay attention to the type of parameters.
4. finish: use this to signal that you have finished all your objectives, args: "response": "final response to let people know you have finished your objectives"

Resources:
1. Internet access for searches and information gathering.
2. Long Term memory management.
3. GPT-3.5 powered Agents for delegation of simple tasks.
4. File output.

Performance Evaluation:
1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

You should only respond in JSON format as described below 
Response Format: 
{
    "thoughts": {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user"
    },
    "command": {
        "name": "command name",
        "args": {
            "arg name": "value"
        }
    }
} 
Ensure the response can be parsed by Python json.loads
System: The current time and date is Wed Apr 26 11:02:25 2023
System: This reminds you of these events from your past:
[]

Human: Determine which next command to use, and respond using the format specified above:

> Finished chain.
{
    "thoughts": {
        "text": "Since the 'get_database_schema' command did not work, I will try to retrieve the schema manually. I will use the following SQL query to retrieve the schema: 'SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;'",
        "reasoning": "If the 'get_database_schema' command is not working, I can manually retrieve the schema using an SQL query. This query will retrieve the names of all the tables and columns in the 'public' schema of the database.",
        "plan": "- Use the SQL query 'SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;' to retrieve the schema\n- Generate SQL query based on given description\n- Use 'select_database_data' command to retrieve query results\n- Count number of rows in query results using Python code",
        "criticism": "I need to make sure that I am using the correct table and column names from the schema in the SQL query. I also need to make sure that I am using the correct syntax for the SQL query.",
        "speak": "I will try to retrieve the schema manually using an SQL query."
    },
    "command": {
        "name": "select_database_data",
        "args": {
            "query": "SELECT table_name, column_name FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;"
        }
    }
}

"""
