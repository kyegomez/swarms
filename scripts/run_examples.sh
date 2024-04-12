#!/bin/bash

# Define a file to keep track of successfully executed scripts
SUCCESS_LOG="successful_runs.log"

for f in /swarms/playground/examples/example_*.py; do
    # Check if the script has been logged as successful
    if grep -Fxq "$f" "$SUCCESS_LOG"; then
        echo "Skipping ${f} as it ran successfully in a previous run."
    else
        # Run the script if not previously successful
        if python "$f" 2>>errors.txt; then
            echo "(${f}) ran successfully without errors."
            # Log the successful script execution
            echo "$f" >>"$SUCCESS_LOG"
        else
            echo "Error encountered in ${f}. Check errors.txt for details."
            break
        fi
    fi
    echo "##############################################################################"
done
