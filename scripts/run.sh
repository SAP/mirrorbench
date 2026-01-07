#!/bin/bash

# Run all benchmark jobs
bash scripts/launch.sh scripts/generate_configs_and_run.py

# Generate comparison reports for each dataset
echo ""
echo "[mirrorbench-script] Generating comparison reports..."
bash scripts/launch.sh scripts/generate_comparison_reports.py