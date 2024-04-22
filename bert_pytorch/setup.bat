@echo off

REM Step 1: Install Python requirements
echo Installing Python requirements...
pip install -r ../requirements.txt

REM Step 2: Run DownloadKinyastory script
echo Running DownloadKinyastory...
python download_kinyastory.py

REM Step 3: Run KinyaTokenizer script
echo Running KinyaTokenizer...
python KinyaTokenizer.py

REM Step 4: Run Kinya_main script
echo Running Kinya_main...
python kinya_main.py --train_dataset tokenized_data.pt --output_path output/bert.model

echo Setup completed successfully.