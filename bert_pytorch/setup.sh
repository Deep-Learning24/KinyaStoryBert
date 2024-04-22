#!/bin/bash

# Step 1: Install Python requirements
echo "Installing Python requirements..."
pip install -r ../requirements.txt

# Step 2: Run DownloadKinyastory script
echo "Running DownloadKinyastory..."
python3 download_kinyastory.py

# Step 3: Run KinyaTokenizer script
echo "Running KinyaTokenizer..."
python3 KinyaTokenizer.py

echo "Setup completed successfully."