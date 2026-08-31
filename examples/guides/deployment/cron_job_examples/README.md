# Cron Job Examples

This directory contains examples demonstrating scheduled task execution using cron jobs.

## Examples

### Start here

- [single_agent_cron.py](single_agent_cron.py) - One agent on one interval, the smallest useful job
- [one_agent_many_tasks_cron.py](one_agent_many_tasks_cron.py) - One agent running several tasks on a shared cadence
- [multi_agent_schedules_cron.py](multi_agent_schedules_cron.py) - Several agents, each on its own cadence, via `CronJob.run_many`
- [non_blocking_cron.py](non_blocking_cron.py) - Start a fleet without blocking, inspect it mid-flight, stop it
- [resilient_cron.py](resilient_cron.py) - Failure handling, error budgets, and live monitoring

### Callbacks and domain examples

- [callback_cron_example.py](callback_cron_example.py) - Cron job with callbacks
- [cron_job_example.py](cron_job_example.py) - Basic cron job example
- [cron_job_figma_stock_swarms_tools_example.py](cron_job_figma_stock_swarms_tools_example.py) - Figma stock swarms tools cron job
- [crypto_concurrent_cron_example.py](crypto_concurrent_cron_example.py) - Concurrent crypto cron job
- [figma_stock_example.py](figma_stock_example.py) - Figma stock example
- [simple_callback_example.py](simple_callback_example.py) - Simple callback example
- [simple_concurrent_crypto_cron.py](simple_concurrent_crypto_cron.py) - Simple concurrent crypto cron
- [solana_price_tracker.py](solana_price_tracker.py) - Solana price tracker cron job

## Overview

A `CronJob` binds one agent to one interval and runs it until you stop it. For several agents on
different cadences, `CronJob.run_many` starts one job per agent so they run together while staying
isolated from each other.

Failures are treated the way cron treats them: a task that raises is logged and retried on the next
tick rather than taking the schedule down. Set `max_consecutive_errors` to stop a job that is failing
every time; when that budget runs out the job stops and `run()` raises, so a dead schedule is never
mistaken for a healthy one. Poll `get_execution_stats()` at any point for successes, failures and the
last error.

Cron job examples demonstrate how to schedule and execute agent tasks on a recurring basis. These examples show various patterns including callback handling, concurrent execution, and domain-specific scheduled tasks like price tracking and stock monitoring.

