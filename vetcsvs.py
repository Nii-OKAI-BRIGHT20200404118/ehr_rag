import csv
from pathlib import Path

def examine_csv_structure(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        print(f"File: {file_path.name}")
        print("Columns:", headers)
        print("Sample data:")
        for _ in range(3):  # Print first 3 rows as samples
            try:
                row = next(reader)
                print(row)
            except StopIteration:
                break
        print("\n")

data_dir = Path('E:\AI RESEEARCH NEW\ehr_rag\Data\syntheaNewData')
for csv_file in data_dir.glob('*.csv'):
    examine_csv_structure(csv_file)