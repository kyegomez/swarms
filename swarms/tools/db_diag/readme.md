# db_diag Tool

Contributor: [Xuanhe Zhou](https://github.com/zhouxh19)

### API Functions

- *obtain_start_and_end_time_of_anomaly*: fetch the time period of an anomaly
- *whether_is_abnormal_metric*: examine whether the values of the input metric appear to be abnormal. //todo: add classic anomaly detection algorithms
- *xxx_diagnosis_agent*: diagnose the root causes of the abnormal metrics in specific region (e.g., memory/cpu problems)


### Setup

1. Follow the steps in [main readme](https://github.com/OpenBMB/swarms.tools/blob/main/README.md)

2. Configure the adopted LLM model in the 84th line of ../../agent/singletool.py, e.g., 

```bash
    self.llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0, openai_api_key=key)
```

3. Modify the settings in *config.ini*, and rename *config.ini* into *my_config.ini*

4. Modify and run the test.py script to test the tool
