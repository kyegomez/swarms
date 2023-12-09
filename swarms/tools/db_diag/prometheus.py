import requests
import json
import datetime
import numpy as np


def prometheus(url, params):
    output = requests.get(url="http://8.131.229.55:9090/" + url, params=params)
    output = output.json()
    print(output)
    # output = json.dumps(res.json())
    output = output["data"]["result"][0]["values"]
    output = np.array([float(value) for _, value in output])
    print(output)
    print(type(output))


if __name__ == "__main__":
    # prometheus('api/v1/query_range', {'query': '100 - (avg(irate(node_cpu_seconds_total{instance=~"123.56.63.105:9100",mode="idle"}[1m])) * 100)', 'start': '1684412385', 'end': '1684412485', 'step': '3'})
    start_timestamp_str = "2023-05-19 22:21:30"
    dt = datetime.datetime.strptime(start_timestamp_str, "%Y-%m-%d %H:%M:%S")
    timestamp = dt.timestamp()
    start_time = timestamp

    end_timestamp_str = "2023-05-19 22:23:30"
    dt = datetime.datetime.strptime(end_timestamp_str, "%Y-%m-%d %H:%M:%S")
    timestamp = dt.timestamp()
    end_time = timestamp

    prometheus(
        "api/v1/query_range",
        {
            "query": 'node_memory_MemTotal_bytes{instance=~"123.56.63.105:9100"} - (node_memory_Cached_bytes{instance=~"123.56.63.105:9100"} + node_memory_Buffers_bytes{instance=~"123.56.63.105:9100"} + node_memory_MemFree_bytes{instance=~"123.56.63.105:9100"})',
            "start": start_time,
            "end": end_time,
            "step": "3",
        },
    )
