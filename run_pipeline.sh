#!/bin/bash

echo "Activating virtual environment..."
source venv/bin/activate

echo "Running big data pipeline..."
python master_script.py

echo "Done!"
