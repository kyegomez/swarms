import json

import json
from difflib import ndiff
import psycopg2
import time

"""
prepare the test samples
"""


def execute_sql(sql):
    conn = psycopg2.connect(
        database="tpch10x", user="xxx", password="xxx", host="xxx", port=xxx
    )

    cur = conn.cursor()
    cur.execute(sql)
    # res = cur.fetchall()[0][0][0]
    res = cur.fetchall()

    conn.commit()
    cur.close()
    conn.close()

    return len(res)


# Load the JSON file as a dictionary
data = {}
with open("text2res_single_table.json", "r") as f:
    data = json.load(f)

# Select only the diverse SQL statements
# Find SQL statements with an edit distance of less than 10
selected_sql = []
for sql1 in data:
    if "sql" in sql1:
        sql1 = sql1["sql"]
        print("==========sql", sql1)
        start_time = time.time()
        res_cnt = execute_sql(sql1)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(res_cnt, elapsed_time)

        selected_sql.append(
            {f"sql": sql1, "res_cnt": res_cnt, "execution_time": elapsed_time}
        )


# Write the dictionary to a JSON file
with open("text2res_single_table2.json", "w") as f:
    json.dump(selected_sql, f)


"""
add text descriptions for queries
"""
if __name__ == "__main__":
    llm = LLM()  # add the def of your llm

    with open("./tpch10x/text2res_single_table2.json", "r") as json_file:
        json_data = json.load(json_file)

    new_json_data = []
    for i, item in enumerate(json_data):
        sql = item["sql"]
        print("========= ", i, sql)
        prompt = (
            "Please convert the following sql query into one natural language sentence: \n"
            + sql
            + "\n Note. 1) Do not mention any other information other than the natural language sentence; 2) Must use the origin table and column names in the sql query."
        )
        text = llm(prompt)
        item["text"] = text
        new_json_data.append(item)
        # print(llm("Describe Shanghai in 200 words."))

    with open("text2res_single_table3.json", "w") as f:
        json.dump(new_json_data, f)


"""
calculate total execution time
"""

with open("text2res_origin.json", "r") as json_file:
    json_data = json.load(json_file)

total_time = 0

for i, item in enumerate(json_data):
    print(item["execution_time"])
    total_time = total_time + float(item["execution_time"])

print(total_time)
