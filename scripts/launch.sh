export PYTHONPATH="${PYTHONPATH}:${pwd}"

uv run $@

# Run any python script with the correct PYTHONPATH set using: bash scripts/launch.sh path/to/script.py