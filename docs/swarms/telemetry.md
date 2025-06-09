# Telemetry Initialization

Swarms runs a small bootup routine whenever the package is imported. The routine configures logging, sets up a workspace directory and silences verbose output from external libraries. This helps reduce noise in development logs and ensures your agents always have a predictable workspace.

## Initialization Steps

1. The [`bootup`](../../swarms/telemetry/bootup.py) function executes automatically during import.
2. `bootup` reads the `SWARMS_VERBOSE_GLOBAL` variable to decide if logging should be suppressed.
3. The WandB library is silenced by setting `WANDB_SILENT` to `"true"`.
4. A workspace directory named `agent_workspace` is created if it does not already exist and the path is exported to `WORKSPACE_DIR`.
5. Deprecation warnings are suppressed and `disable_logging()` sets `TF_CPP_MIN_LOG_LEVEL` to `"3"`.

This initialization ensures minimal console output while still capturing important telemetry about your system.

## Enabling or Disabling Telemetry

Telemetry collection is controlled with the environment variable `USE_TELEMETRY`.

```bash
# Enable telemetry
USE_TELEMETRY=true

# Disable telemetry
USE_TELEMETRY=false
```

When disabled, metrics from the `swarms.telemetry` package are not sent to the analytics endpoint.

## Environment Variables Set on Bootup

| Variable | Default | Purpose |
|----------|---------|---------|
| `SWARMS_VERBOSE_GLOBAL` | `False` | Controls the verbosity of logging during startup. |
| `WANDB_SILENT` | `true` | Prevents the WandB library from printing to stdout. |
| `WORKSPACE_DIR` | `./agent_workspace` | Path to the working directory used by agents. |
| `TF_CPP_MIN_LOG_LEVEL` | `3` | Hides TensorFlow warnings when logging is disabled. |

These variables are set automatically each time the package loads so you rarely need to set them manually. `SWARMS_VERBOSE_GLOBAL` is read before bootup to decide whether to disable logging. All others are created or overwritten when the initialization runs.

---

For more details, review the [`bootup` implementation](../../swarms/telemetry/bootup.py) and the [`disable_logging`](../../swarms/utils/disable_logging.py) helper.
