#!/bin/bash

set -euo pipefail

if [ -f ".venv/bin/activate" ]; then
  echo "Activating virtual environment from .venv/..."
  source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
  echo "Activating virtual environment from venv/..."
  source venv/bin/activate
else
  echo "No virtual environment found in .venv/ or venv/."
  exit 1
fi

echo "Running big data pipeline..."
python master_script.py

echo "Done!"
