#!/bin/bash

total_size=0

while read -r package; do
    # Fetch package info from PyPI
    size=$(curl -s "https://pypi.org/pypi/$package/json" | jq '.releases | to_entries | map(.value[].size) | add')
    if [ "$size" != "null" ]; then
        echo "$package: $(echo "scale=2; $size/1024/1024" | bc) MB"
        total_size=$(echo "$total_size + $size" | bc)
    else
        echo "$package: Not found"
    fi
done <requirements.txt

echo "Total size: $(echo "scale=2; $total_size/1024/1024" | bc) MB"
