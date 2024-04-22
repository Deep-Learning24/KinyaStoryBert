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

# create output directory if it doesn't exist

if [ ! -d "output" ]; then
  mkdir output
fi

# Step 4: Run Kinya_main script
echo Running Kinya_main...
python3 kinya_main.py --train_dataset tokenized_data.pt --output_path output/bert.model

# create output directory if it doesn't exist

if [ ! -d "output" ]; then
  mkdir output
fi

echo "Setup completed successfully."