import csv
import json
from pathlib import Path
from typing import Dict
import shutil

class SyntheaPreprocessor:
    def __init__(self, data_directory: str, output_directory: str):
        self.data_dir = Path('E:/AI RESEEARCH NEW/ehr_rag/Data/synthea_sample_data_csv_apr2020/csv')
        self.output_dir = Path('E:/AI RESEEARCH NEW/ehr_rag/Data/syntheaNewData')
        self.id_mapping = {}
        self.current_id = 1

    def get_short_id(self, long_id: str) -> str:
        if long_id not in self.id_mapping:
            self.id_mapping[long_id] = f"P{self.current_id:04d}"
            self.current_id += 1
        return self.id_mapping[long_id]

    def create_id_mapping(self):
        patients_file = self.data_dir / 'patients.csv'
        with open(patients_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.get_short_id(row['Id'])

    def update_csv_files(self):
        csv_files = ['patients.csv', 'encounters.csv', 'conditions.csv', 'medications.csv', 'procedures.csv']
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for file_name in csv_files:
            input_file = self.data_dir / file_name
            output_file = self.output_dir / file_name
            if not input_file.exists():
                print(f"File {file_name} not found. Skipping.")
                continue

            try:
                with open(input_file, 'r', encoding='utf-8') as infile, \
                     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                    reader = csv.DictReader(infile)
                    fieldnames = reader.fieldnames

                    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                    writer.writeheader()

                    id_field = 'Id' if file_name == 'patients.csv' else 'PATIENT'
                    for row in reader:
                        if id_field in row:
                            row[id_field] = self.get_short_id(row[id_field])
                        else:
                            print(f"Warning: '{id_field}' column not found in {file_name}")
                        writer.writerow(row)

                print(f"Updated {file_name} with short IDs.")
            except UnicodeDecodeError as e:
                print(f"Error processing {file_name}: {e}")
                print("Try manually opening and resaving the file as UTF-8 in a text editor.")

    def save_id_mapping(self):
        with open(self.output_dir / 'id_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(self.id_mapping, f, indent=2, ensure_ascii=False)
        print("Saved ID mapping to id_mapping.json in the output directory")

    def preprocess(self):
        self.create_id_mapping()
        self.update_csv_files()
        self.save_id_mapping()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python synthea_preprocessor.py <path_to_synthea_output>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_dir = Path(data_dir) / "preprocessed_output"
    preprocessor = SyntheaPreprocessor(data_dir, str(output_dir))
    preprocessor.preprocess()
    print(f"Preprocessing complete. New CSV files with short IDs have been created in {output_dir}")